import json
import pandas as pd
import numpy as np
from dates import target_holidays, Tenor, day_count_fraction, BusinessDay
from typing import TypeAlias
from dateutil.relativedelta import relativedelta as tdelta
from scipy.interpolate import interp1d
from scipy.optimize import brentq

# Type aliases
Date: TypeAlias = pd.Timestamp
Rate: TypeAlias = float
Df:   TypeAlias = pd.DataFrame

# Bloomberg instrument conventions, loaded once at module level
CURVE_CONVENTIONS = {}
with open("market_conventions/estr.json") as f:
    CURVE_CONVENTIONS["ESTR"] = json.load(f)
with open("market_conventions/euribor6m.json") as f:
    CURVE_CONVENTIONS["EURIBOR6M"] = json.load(f)

# Par rates and pre-computed zero curves, indexed for fast lookup
rates      = pd.read_parquet("clean_data/rates.parquet").set_index(["Date", "Curve", "Tenor"])
zero_rates = pd.read_parquet("clean_data/zero_rates.parquet").set_index(["TradeDate", "Curve", "Tenor"])

# TARGET calendar covering the full dataset range plus a 31-year buffer
calendar = target_holidays(
    rates.index.get_level_values(0).min().year,
    rates.index.get_level_values(0).max().year + 31
)
buscal = np.busdaycalendar(holidays=calendar)


def schedule_generation(
        trade_date: Date,
        tenor: Tenor,
        frequency: str = "Annual",
        fixing_lag: str = "2 Business Days",
        pay_delay: str  = "1 Business Day",
        cal: list = buscal
    ) -> pd.DataFrame:
    """
    Generate a payment schedule for a swap or cap leg.
    Periods are built backward from maturity to avoid stub issues.
    Vectorized business day adjustment via np.busday_offset.
    Returns: DataFrame with FixingDate, AccrualStart, AccrualEnd, PaymentDate.
    """
    trade_date  = BusinessDay(trade_date, calendar=cal)
    settle_date = trade_date.shift("2D", "following")
    tenor       = Tenor(tenor) if isinstance(tenor, str) else tenor
    freq_months = 12 if frequency == "Annual" else 6
    n_coupons   = int(np.ceil(tenor.months / freq_months))
    maturity    = settle_date.shift(tenor, "modifiedfollowing")
    pay_delay   = int(pay_delay[0])
    fix_lag     = -int(fixing_lag[0])

    # Build period end dates by stepping backward from maturity
    period_end = []
    for i in range(n_coupons):
        shift = tdelta(months=i * freq_months)
        period_end.append(maturity.date - shift)

    period_end   = np.busday_offset(sorted(period_end), 0, "modifiedfollowing", busdaycal=cal)
    period_start = np.concatenate([[np.datetime64(settle_date.date, "D")], period_end[:-1]])
    pay_date     = np.busday_offset(period_end,   pay_delay, "modifiedfollowing", busdaycal=cal)
    fix_date     = np.busday_offset(period_start, fix_lag,   "modifiedfollowing", busdaycal=cal)

    return pd.DataFrame(
        np.column_stack((fix_date, period_start, period_end, pay_date)),
        columns=["FixingDate", "AccrualStart", "AccrualEnd", "PaymentDate"]
    )


# =============================================================================
# Instrument hierarchy
# Instrument    base class: tenor, rate, maturity from T+2 settlement
# SpotInstrument    1D and 2D: matures from trade date (Following)
# DepositInstrument 1W-11M:   matures from settlement (Modified Following)
# SwapInstrument    1Y+:      generates fixed and (for Euribor) float schedules
# =============================================================================

class Instrument:
    def __init__(self, trade_date: Date, tenor: Tenor, rate: Rate, curve: str):
        self.trade_date = BusinessDay(trade_date, calendar=buscal)
        self.tenor      = Tenor(tenor) if isinstance(tenor, str) else tenor
        self.rate       = rate / 100
        self.curve      = curve
        self.maturity   = self.trade_date.shift("2D").shift(tenor, "modifiedfollowing").date

    def __repr__(self):
        return f"Instrument({self.tenor.tenor}, {self.rate:.4%}, {self.maturity})"


class SpotInstrument(Instrument):
    """Overnight and spot instruments: mature from trade date, Following convention."""
    def __init__(self, trade_date, tenor, rate, curve):
        super().__init__(trade_date, tenor, rate, curve)
        self.maturity = self.trade_date.shift(tenor, "following").date

    def __repr__(self):
        return f"Spot{super().__repr__()}"


class DepositInstrument(Instrument):
    """Sub-1Y deposits: mature from settlement date (T+2), Modified Following."""
    def __init__(self, trade_date, tenor, rate, curve):
        super().__init__(trade_date, tenor, rate, curve)

    def __repr__(self):
        return f"Deposit{super().__repr__()}"


class SwapInstrument(Instrument):
    """
    1Y+ swap instruments. Conventions read from CURVE_CONVENTIONS JSON.
    ESTR: fixed schedule only (OIS float leg is handled analytically).
    EURIBOR6M: both fixed and float schedules generated.
    """
    def __init__(self, trade_date, tenor, rate, curve):
        super().__init__(trade_date, tenor, rate, curve)
        tenor      = Tenor(tenor) if isinstance(tenor, str) else tenor
        self.tenor = tenor
        self.rate  = rate / 100

        fixed_leg = CURVE_CONVENTIONS[curve]["fixed_leg"]
        float_leg = CURVE_CONVENTIONS[curve]["float_leg"]

        self.fixed_schedule = schedule_generation(
            trade_date, self.tenor,
            fixed_leg["pay_freq"], "0D", fixed_leg["pay_delay"]
        ).drop("FixingDate", axis=1)

        if curve == "EURIBOR6M":
            self.float_schedule = schedule_generation(
                trade_date, self.tenor,
                float_leg["pay_freq"], float_leg["fixing_lag"], float_leg["pay_delay"]
            )

    def __repr__(self):
        return f"Swap{super().__repr__()}"


class ParCurve:
    """
    Market par rate curve for a given trade date and curve.
    Classifies each tenor into the appropriate instrument type and
    sorts all instruments by maturity (required for sequential bootstrapping).
    """
    def __init__(self, trade_date: Date, curve: str):
        self.trade_date  = pd.to_datetime(trade_date)
        self.curve       = curve
        self.__par_curve = rates.xs(level=("Curve", "Date"), key=(curve, trade_date))
        self.__load()

    def __load(self) -> None:
        instruments = {}
        for tenor in self.__par_curve.index:
            rate = self.__par_curve.loc[tenor].Rate
            if Tenor(tenor) == Tenor("1D"):
                instruments[tenor] = SpotInstrument(self.trade_date, tenor, rate, self.curve)
                # Synthetic 2D point: anchors the discount curve at settlement date
                instruments["2D"]  = SpotInstrument(self.trade_date, "2D", rate, self.curve)
            elif Tenor(tenor) < Tenor("1Y"):
                instruments[tenor] = DepositInstrument(self.trade_date, tenor, rate, self.curve)
            else:
                instruments[tenor] = SwapInstrument(self.trade_date, tenor, rate, self.curve)

        self.instruments = dict(sorted(instruments.items(), key=lambda x: x[1].maturity))

    def output(self) -> Df:
        instruments = list(self.instruments.values())
        df = pd.DataFrame({
            "Tenor":    [i.tenor.tenor for i in instruments],
            "Maturity": [i.maturity    for i in instruments],
            "Rate":     [i.rate        for i in instruments],
        })
        df["Curve"]     = self.curve
        df["TradeDate"] = self.trade_date
        df.Tenor        = df.Tenor.astype("category")
        df.Curve        = df.Curve.astype("category")
        df.Maturity     = pd.to_datetime(df.Maturity)
        return df[["TradeDate", "Curve", "Tenor", "Maturity", "Rate"]]

    def __getitem__(self, tenor: str):
        return self.instruments[tenor]

    def __repr__(self):
        return f"{self.curve}({self.trade_date.date()})"


class ZeroCurve:
    """
    Discount curve built from bootstrapped knot points.
    Interpolation: log-linear on discount factors (guarantees positivity
    and monotonicity). Extrapolation: flat forward beyond last knot.
    Anchor point P(t0, t0) = 1.0 always prepended.
    """
    def __init__(self, trade_date: Date, tenors, maturities, dfs, curve: str, conv="ACT/360"):
        self.trade_date   = pd.to_datetime(trade_date)
        self.curve_points = np.array(list(zip(tenors, maturities, dfs)))
        # Prepend anchor: P(t0, t0) = 1
        self.curve_points = np.concatenate((
            np.array([[np.str_("0D"), self.trade_date, np.float64(1.0)]]),
            self.curve_points
        ))
        self.__sort()

        self.maturities = self.curve_points.T[1]
        self.dfs        = np.array(self.curve_points.T[2]).astype(float)
        self.offsets    = np.array([(i - self.trade_date).days for i in self.maturities])
        self.tenors     = self.curve_points.T[0]
        self.curve      = curve
        self.conv       = conv

    def __sort(self):
        self.curve_points = np.array(sorted(self.curve_points, key=lambda x: x[1]))

    def interpolator(self):
        """Log-linear interpolator on discount factors."""
        interp = interp1d(self.offsets, np.log(self.dfs), kind="linear", fill_value="extrapolate")
        return lambda t: np.exp(interp(t))

    def discount(self, t) -> float:
        """Return P(t0, t) for any date t."""
        t = (pd.to_datetime(t) - self.trade_date).days
        return self.interpolator()(t)

    def continuous_rate(self, t) -> float:
        """Continuously compounded zero rate to date t."""
        tau = day_count_fraction(self.trade_date, t, self.conv)
        return -np.log(self.discount(t)) / tau

    def forward_rate(self, start: Date, end: Date) -> float:
        """Simply compounded forward rate F(start, end) using ACT/360."""
        tau = day_count_fraction(start, end, self.conv)
        return (self.discount(start) / self.discount(end) - 1) / tau

    def inst_fwd_rate(self, t):
        """
        Instantaneous forward rate via central finite difference on log P.
        f(t) = -d/dt log P(t0, t), approximated with a 1-day bump.
        """
        e   = 1  # 1-day bump
        t_u = np.array([i + tdelta(days=e)  for i in t])
        t_d = np.array([i + tdelta(days=-e) for i in t])
        u   = np.log(self.discount(t_u))
        d   = np.log(self.discount(t_d))
        return -(u - d) / (2 * e / int(self.conv.split("/")[-1]))

    def time(self):
        """Daily date range from first to last knot point."""
        return pd.date_range(self.maturities.min(), self.maturities.max())

    def output(self) -> Df:
        df = pd.DataFrame({
            "Maturity":       self.maturities,
            "Offset":         self.offsets,
            "DiscountFactor": self.dfs,
            "Tenor":          self.tenors,
        })
        df["Curve"]     = self.curve
        df["TradeDate"] = self.trade_date
        df.Curve        = df.Curve.astype("category")
        df.Tenor        = df.Tenor.astype("category")
        df.Offset       = pd.to_numeric(df.Offset)
        df.DiscountFactor = pd.to_numeric(df.DiscountFactor)
        return df[["TradeDate", "Curve", "Tenor", "Maturity", "Offset", "DiscountFactor"]]

    def __getitem__(self, name) -> float:
        return self.discount(name)

    def __repr__(self):
        return f"ZeroCurve({self.trade_date.date()})"


class Bootstrapper:
    """
    Sequential bootstrap of a discount curve from par market instruments.

    ESTR (discount_curve=None):
        - Spot/deposit: direct formula df = 1/(1 + r*tau)
        - OIS swap: analytical inversion of the par swap equation using
          the OIS identity PV(float) = P(T_start) - P(T_end)

    EURIBOR6M (discount_curve=ZeroCurve):
        - Multi-curve: cashflows discounted with ESTR curve
        - Forward rates from the Euribor curve being built
        - Last caplet period solved numerically via brentq (nonlinear)

    Bootstrapping stops at 12Y due to missing intermediate knot points
    beyond that tenor (13Y, 14Y not quoted).
    """
    def __init__(self, par_curve: ParCurve, discount_curve: ZeroCurve = None):
        self.par_curve      = par_curve
        self.discount_curve = discount_curve
        self.trade_date     = par_curve.trade_date
        self.settle_date    = BusinessDay(self.trade_date, buscal).shift("2D").date
        self.z_curve        = {}
        self.float_conv     = CURVE_CONVENTIONS[self.par_curve.curve]["float_leg"]["day_count"]
        self.fixed_conv     = CURVE_CONVENTIONS[self.par_curve.curve]["fixed_leg"]["day_count"]
        self.df_settle      = self.bootstrap_spot(par_curve["2D"])
        self.__interp_cache = None
        self.run()

    def run(self) -> None:
        for tenor, instrument in self.par_curve.instruments.items():
            if Tenor(tenor) > Tenor("12Y"):
                break
            if isinstance(instrument, SpotInstrument):
                df = self.bootstrap_spot(instrument)
            elif isinstance(instrument, DepositInstrument):
                df = self.bootstrap_deposit(instrument)
            else:
                df = self.bootstrap_ois_swap(instrument) if self.discount_curve is None \
                     else self.bootstrap_irs_swap(instrument)

            self.z_curve[(instrument.maturity, tenor)] = df
            self.__interp_cache = None  # invalidate on each new knot point

    def bootstrap_spot(self, instrument: Instrument) -> float:
        """P(t0, T) = 1 / (1 + r * tau), measured from trade date."""
        tau = day_count_fraction(self.trade_date, instrument.maturity, self.float_conv)
        return 1 / (1 + instrument.rate * tau)

    def bootstrap_deposit(self, instrument: Instrument) -> float:
        """P(t0, T) = P(t0, T_settle) * 1 / (1 + r * tau), tau from settlement."""
        tau = day_count_fraction(self.settle_date, instrument.maturity, self.float_conv)
        return 1 / (1 + instrument.rate * tau) * self.df_settle

    def bootstrap_ois_swap(self, instrument: Instrument) -> float:
        """
        Analytical OIS bootstrap. OIS identity: PV(float) = 1 - P(T_N).
        P(T_N) = (1 - r * annuity) / (1 + r * tau_last)
        where annuity = sum of tau_i * P(T_i) over all previous coupon dates.
        """
        schedule = instrument.fixed_schedule
        rate     = instrument.rate

        annuity = sum(
            day_count_fraction(row.AccrualStart, row.AccrualEnd, self.float_conv)
            * self.__get_df(row.AccrualEnd)
            for _, row in schedule.iloc[:-1].iterrows()
        )

        last     = schedule.iloc[-1]
        tau_last = day_count_fraction(last.AccrualStart, last.AccrualEnd, self.float_conv)
        df       = (1 - rate * annuity) / (1 + rate * tau_last)
        return df * self.df_settle

    def bootstrap_irs_swap(self, instrument: Instrument) -> float:
        """
        Multi-curve IRS bootstrap. Fixed leg uses 30U/360 discounted with ESTR.
        Float leg uses ACT/360 with Euribor forward rates from the curve being built.
        Last float period contains the unknown P_EUR(T_N) nonlinearly (via forward
        rate definition), solved with brentq.
        """
        fixed_schedule = instrument.fixed_schedule
        float_schedule = instrument.float_schedule
        rate           = instrument.rate

        # Fixed leg annuity (all periods except last)
        fixed_annuity = sum(
            day_count_fraction(row.AccrualStart, row.AccrualEnd, self.fixed_conv)
            * self.discount_curve.discount(row.PaymentDate)
            for _, row in fixed_schedule.iloc[:-1].iterrows()
        )
        last_fixed     = fixed_schedule.iloc[-1]
        tau_last_fixed = day_count_fraction(last_fixed.AccrualStart, last_fixed.AccrualEnd, self.fixed_conv)
        df_last_fixed  = self.discount_curve.discount(last_fixed.PaymentDate)

        # Float leg PV (all periods except last, forward rates from Euribor curve)
        float_pv = 0.0
        for _, row in float_schedule.iloc[:-1].iterrows():
            tau      = day_count_fraction(row.AccrualStart, row.AccrualEnd, self.float_conv)
            df_start = self.__get_df(row.AccrualStart)
            df_end   = self.__get_df(row.AccrualEnd)
            fwd      = (df_start / df_end - 1) / tau
            df_pay   = self.discount_curve.discount(row.PaymentDate)
            float_pv += tau * fwd * df_pay

        # Last float period: P_EUR(T_N) appears in the forward rate -> nonlinear
        last_float     = float_schedule.iloc[-1]
        tau_last_float = day_count_fraction(last_float.AccrualStart, last_float.AccrualEnd, self.float_conv)
        df_start_last  = self.__get_df(last_float.AccrualStart)
        df_pay_last    = self.discount_curve.discount(last_float.PaymentDate)

        def equation(x):
            fwd_last   = (df_start_last / x - 1) / tau_last_float
            float_last = tau_last_float * fwd_last * df_pay_last
            return rate * (fixed_annuity + tau_last_fixed * df_last_fixed) - (float_pv + float_last)

        return brentq(equation, 1e-6, 2.0)

    def __get_df(self, date: Date) -> float:
        """Log-linear interpolation on the curve being built. Cache invalidated on each new knot."""
        if self.__interp_cache is None:
            x = np.array([(i[0] - self.trade_date).days for i in self.z_curve])
            y = np.log(np.array(list(self.z_curve.values())))
            self.__interp_cache = interp1d(x, y, kind="linear", fill_value="extrapolate")
        target = (np.datetime64(pd.to_datetime(date)) - self.trade_date).days
        return np.exp(self.__interp_cache(target))

    def build(self) -> ZeroCurve:
        """Return a ZeroCurve object from the bootstrapped knot points."""
        maturities = np.array([i[0] for i in self.z_curve])
        dfs        = np.array(list(self.z_curve.values()))
        return ZeroCurve(self.trade_date, maturities, dfs, self.par_curve.curve)

    def output(self) -> Df:
        """Return bootstrapped curve as a long-format DataFrame for storage."""
        df = pd.DataFrame({
            "Maturity":       [i[0] for i in self.z_curve],
            "Tenor":          [i[1] for i in self.z_curve],
            "DiscountFactor": list(self.z_curve.values()),
        })
        df["Curve"]     = self.par_curve.curve
        df["TradeDate"] = self.trade_date
        df.Curve        = df.Curve.astype("category")
        df.Tenor        = df.Tenor.astype("category")
        return df[["TradeDate", "Curve", "Tenor", "Maturity", "DiscountFactor"]]

    def __repr__(self):
        disc = "" if self.discount_curve is None else f", discount_curve={self.discount_curve.curve}"
        return f"Bootstrapper({self.trade_date.date()}, par_curve={self.par_curve.curve}{disc})"


# =============================================================================
# Incremental update loops — append only new dates to zero_rates.parquet
# =============================================================================

def bootstrap_estr_loop():
    """Bootstrap ESTR OIS curves for all dates not yet in zero_rates.parquet."""
    base  = pd.read_parquet("clean_data/zero_rates.parquet")
    dates = sorted(
        rates.reset_index()[
            rates.reset_index().Date > base[base.Curve == "ESTR"].TradeDate.max()
        ].Date.unique()
    )
    new = []
    for date in dates:
        try:
            new.append(Bootstrapper(ParCurve(date, "ESTR")).output())
        except:
            continue

    if new:
        pd.concat([base] + new)\
            .sort_values(["TradeDate", "Curve", "Maturity"])\
            .reset_index(drop=True)\
            .to_parquet("clean_data/zero_rates.parquet", index=False)
        print("Updated correctly.")
    else:
        print("No new data.")


def bootstrap_euribor_loop():
    """Bootstrap Euribor 6M curves for all dates not yet in zero_rates.parquet.
    Requires ESTR curve to be bootstrapped first on each date."""
    base  = pd.read_parquet("clean_data/zero_rates.parquet")
    dates = sorted(
        rates.reset_index()[
            rates.reset_index().Date > base[base.Curve == "EURIBOR6M"].TradeDate.max()
        ].Date.unique()
    )
    new = []
    for date in dates:
        try:
            disc = Bootstrapper(ParCurve(date, "ESTR")).build()
            new.append(Bootstrapper(ParCurve(date, "EURIBOR6M"), disc).output())
            print(f"{date.date()}: OK")
        except:
            continue

    if new:
        pd.concat([base] + new)\
            .sort_values(["TradeDate", "Curve", "Maturity"])\
            .reset_index(drop=True)\
            .to_parquet("clean_data/zero_rates.parquet", index=False)
        print("Updated correctly.")
    else:
        print("No new data.")


def zero_curve_load(trade_date, curve) -> ZeroCurve:
    """Load a pre-computed ZeroCurve from the Parquet store for a given date and curve."""
    trade_date = np.datetime64(pd.to_datetime(trade_date))
    z          = zero_rates.xs((trade_date, curve))
    conv       = CURVE_CONVENTIONS[curve]["float_leg"]["day_count"]
    return ZeroCurve(trade_date, z.index.tolist(), z.Maturity.tolist(), z.DiscountFactor.tolist(), curve, conv)
