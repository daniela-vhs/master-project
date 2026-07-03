import json
import pandas as pd
import numpy as np
from dates import target_holidays, Tenor, day_count_fraction, BusinessDay
from typing import TypeAlias
from dateutil.relativedelta import relativedelta as tdelta
from scipy.interpolate import interp1d
from scipy.optimize import brentq

# Type Aliases
Date:  TypeAlias = pd.Timestamp
Rate:  TypeAlias = float
Df:    TypeAlias = pd.DataFrame

# Curve Definitions
CURVE_CONVENTIONS = dict()

with open("market_conventions/estr.json") as f:
    CURVE_CONVENTIONS["ESTR"] = json.load(f)

with open("market_conventions/euribor6m.json") as f:
    CURVE_CONVENTIONS["EURIBOR6M"] = json.load(f)

# Data
rates      = pd.read_parquet("clean_data/rates.parquet").set_index(["Date", "Curve", "Tenor"])
zero_rates = pd.read_parquet("clean_data/zero_rates.parquet").set_index(["TradeDate", "Curve", "Tenor"])
calendar   = target_holidays(rates.index.get_level_values(0).min().year, rates.index.get_level_values(0).max().year + 31)
buscal     = np.busdaycalendar(holidays=calendar)

def schedule_generation(
        trade_date: Date,
        tenor: Tenor,
        frequency: str="Annual",
        fixing_lag: str="2 Business Days",
        pay_delay: str="1 Business Day",
        cal: list=buscal
    ):
    
    trade_date  = BusinessDay(trade_date, calendar=cal)
    settle_date = trade_date.shift("2D", "following")
    tenor       = Tenor(tenor) if isinstance(tenor, str) else tenor
    freq_months = 12 if frequency == "Annual" else 6
    n_coupons   = int(np.ceil(tenor.months / freq_months))
    maturity    = settle_date.shift(tenor, "modifiedfollowing")
    pay_delay   = int(pay_delay[0])
    fix_lag     = -int(fixing_lag[0])

    period_end = []
    i = 0

    while i < n_coupons:
        shift = tdelta(months=i * (12 if frequency == "Annual" else 6))
        period_end.append(maturity.date - shift)
        i += 1

    period_end.sort()
    period_end   = np.busday_offset(period_end, 0, "modifiedfollowing", busdaycal=cal)
    period_start = np.concatenate([[np.datetime64(settle_date.date, 'D')], period_end[:-1]])
    pay_date     = np.busday_offset(period_end, pay_delay, "modifiedfollowing", busdaycal=cal)
    fix_date     = np.busday_offset(period_start, fix_lag, "modifiedfollowing", busdaycal=cal)

    calendar = pd.DataFrame(np.column_stack((fix_date, period_start, period_end, pay_date)), columns=["FixingDate", "AccrualStart", "AccrualEnd", "PaymentDate"])

    return calendar

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
    def __init__(self, trade_date: Date, tenor: Tenor, rate: Rate, curve: str):
        super().__init__(trade_date, tenor, rate, curve)
        self.maturity = self.trade_date.shift(tenor, "following").date

    def __repr__(self):
        return f"Spot{super().__repr__()}"

class DepositInstrument(Instrument):
    def __init__(self, trade_date: Date, tenor: Tenor, rate: Rate, curve: str):
        super().__init__(trade_date, tenor, rate, curve)

    def __repr__(self):
        return f"Deposit{super().__repr__()}"

class SwapInstrument(Instrument):
    def __init__(self, trade_date: Date, tenor: Tenor, rate: Rate, curve: str):
        super().__init__(trade_date, tenor, rate, curve)
        tenor      = Tenor(tenor) if isinstance(tenor, str) else tenor
        self.tenor = tenor
        self.rate  = rate / 100

        fixed_leg = CURVE_CONVENTIONS[curve]["fixed_leg"]
        float_leg = CURVE_CONVENTIONS[curve]["float_leg"]

        self.fixed_schedule = schedule_generation(trade_date,
                                                  self.tenor,
                                                  fixed_leg["pay_freq"],
                                                  "0D",
                                                  fixed_leg["pay_delay"],
                                                  cal=buscal
                                                  ).drop("FixingDate", axis=1)
        
        if curve == "EURIBOR6M":
            self.float_schedule = schedule_generation(trade_date,
                                                    self.tenor,
                                                    float_leg["pay_freq"],
                                                    float_leg["fixing_lag"],
                                                    float_leg["pay_delay"],
                                                    cal=buscal
                                                    )

    def __repr__(self):
        return f"Swap{super().__repr__()}"

class ParCurve:
    def __init__(self, trade_date: Date, curve: str):
        self.trade_date  = pd.to_datetime(trade_date)
        self.curve       = curve
        self.__par_curve = rates.xs(level=("Curve", "Date"), key=(curve, trade_date))
        self.__load()

    def __load(self) -> None:
        instruments = dict()

        for tenor in self.__par_curve.index:
            if Tenor(tenor) == Tenor("1D"):
                instruments[tenor] = SpotInstrument(self.trade_date, tenor, self.__par_curve.loc[tenor].Rate, self.curve)
                instruments["2D"] = SpotInstrument(self.trade_date, "2D", self.__par_curve.loc[tenor].Rate, self.curve)

            elif Tenor(tenor) < Tenor("1Y"):
                instruments[tenor] = DepositInstrument(self.trade_date, tenor, self.__par_curve.loc[tenor].Rate, self.curve)
            else:
                instruments[tenor] = SwapInstrument(self.trade_date, tenor, self.__par_curve.loc[tenor].Rate, self.curve)

        instruments = dict(sorted(instruments.items(), key=lambda x: x[1].maturity))

        self.instruments = instruments

    def output(self):
        data = self.instruments
        
        instruments = list(data.values())
        tenors      = np.array(list(data)).astype("object")
        maturities  = np.array([i.maturity for i in instruments])
        rates       = [i.rate for i in instruments]
        
        df              = pd.DataFrame(np.column_stack((tenors, maturities, rates)), columns=("Tenor", "Maturity", "Rate"))
        df["Curve"]     = self.curve
        df["TradeDate"] = self.trade_date

        df.Tenor    = df.Tenor.astype("category")
        df.Rate     = pd.to_numeric(df.Rate)
        df.Maturity = pd.to_datetime(df.Maturity)
        df.Curve    = df.Curve.astype("category")
        return df[["TradeDate", "Curve", "Tenor", "Maturity", "Rate"]]

    def __getitem__(self, tenor: str) -> SwapInstrument:
        return self.instruments[tenor]

    def __repr__(self) -> str:
        return f"{self.curve}({self.trade_date.date()})"

class ZeroCurve:
    def __init__(self, trade_date: Date, tenors: list[Tenor], maturities: list[Date], dfs: list[float], curve: str, conv="ACT/360"):
        self.trade_date   = pd.to_datetime(trade_date)
        self.curve_points = np.array(list(zip(tenors, maturities, dfs)))
        self.curve_points = np.concatenate((np.array([[np.str_("0D"), self.trade_date, np.float64(1.0)]]), self.curve_points))
        self.__sort()

        self.maturities = self.curve_points.T[1]
        self.dfs        = np.array(self.curve_points.T[2]).astype(float)
        self.offsets    = np.array([(i - self.trade_date).days for i in self.maturities])
        self.tenors     = self.curve_points.T[0]

        self.curve = curve
        self.conv  = conv

    def __sort(self):
        self.curve_points = np.array(sorted(self.curve_points, key=lambda x: x[1]))

    def interpolator(self):
        # Maturities
        x = self.offsets

        # Log Discount Factors
        y = np.log(self.dfs)

        interpolator = interp1d(x, y, kind="linear", fill_value="extrapolate")

        return lambda t: np.exp(interpolator(t))
    
    def discount(self, t):
        # Get Discount Factor
        t = (pd.to_datetime(t) - self.trade_date).days
        interpolator = self.interpolator()
        return interpolator(t)
    
    def continuous_rate(self, t):
        # Get Rate
        day_fraction = day_count_fraction(self.trade_date, t, self.conv)
        return -np.log(self.discount(t)) / day_fraction
    
    def forward_rate(self, start: Date, end: Date) -> float:
        df_start = self[start]
        df_end   = self[end]
        tau      = day_count_fraction(start, end, self.conv)
        return (df_start / df_end - 1) / tau
    
    def inst_fwd_rate(self, t):
        # Get instantaneous forward rate
        e = 1
        t_u = np.array([i + tdelta(days=e) for i in t])
        t_d = np.array([i + tdelta(days=-e) for i in t])
    
        u = np.log(self.discount(t_u))
        d = np.log(self.discount(t_d))

        return -(u - d) / (2 * e / int(self.conv.split("/")[-1]))
    
    def time(self):
        # Helper
        min_date = self.maturities.min()
        max_date = self.maturities.max()
        return pd.date_range(min_date, max_date)
    
    def output(self):
        df = pd.DataFrame(np.column_stack((self.maturities, self.offsets, self.dfs)), columns=["Maturity", "Offset", "DiscountFactor"])
        
        df["Curve"]     = self.curve
        df["TradeDate"] = self.trade_date
        df["Tenor"]     = self.tenors

        df.Curve = df.Curve.astype("category")
        df.Tenor = df.Tenor.astype("category")
        df.Offset         = pd.to_numeric(df.Offset)
        df.DiscountFactor = pd.to_numeric(df.DiscountFactor)
        
        df = df[["TradeDate", "Curve", "Tenor", "Maturity", "Offset", "DiscountFactor"]]
        return df
    
    def __getitem__(self, name) -> float:
        return self.discount(name)
    
    def __repr__(self):
        return f"ZeroCurve({str(self.trade_date.date())})"

class Bootstrapper:
    def __init__(self,
                 par_curve: ParCurve,
                 discount_curve: ZeroCurve = None):
        self.par_curve      = par_curve
        self.discount_curve = discount_curve
        self.trade_date     = par_curve.trade_date
        self.settle_date    = BusinessDay(self.trade_date, buscal).shift("2D").date
        self.z_curve        = dict()
        self.float_conv     = CURVE_CONVENTIONS[self.par_curve.curve]["float_leg"]["day_count"]
        self.fixed_conv     = CURVE_CONVENTIONS[self.par_curve.curve]["fixed_leg"]["day_count"]
        self.df_settle      = self.bootstrap_spot(par_curve["2D"])
        self.run()
        self.__interp_cache = None

    def run(self) -> None:
        for tenor, instrument in self.par_curve.instruments.items():
            if Tenor(tenor) > Tenor("12Y"):
                break
            
            if isinstance(instrument, SpotInstrument):
                df = self.bootstrap_spot(instrument)

            elif isinstance(instrument, DepositInstrument):
                df = self.bootstrap_deposit(instrument)

            else:
                if self.discount_curve is None:
                    df = self.bootstrap_ois_swap(instrument)
                else:
                    df = self.bootstrap_irs_swap(instrument)
                
            self.z_curve[(instrument.maturity, tenor)] = df
            self.__interp_cache = None

        return self.par_curve.instruments

    def bootstrap_spot(self, instrument: Instrument) -> float:
        conv = self.float_conv
        tau  = day_count_fraction(self.trade_date, instrument.maturity, conv)
        return 1 / (1 + instrument.rate * tau)

    def bootstrap_deposit(self, instrument: Instrument) -> float:
        conv = self.float_conv
        tau  = day_count_fraction(self.settle_date, instrument.maturity, conv)
        return 1 / (1 + instrument.rate * tau) * self.df_settle
    
    def bootstrap_ois_swap(self, instrument: Instrument) -> float:
        schedule = instrument.fixed_schedule
        rate     = instrument.rate

        # Annuity
        annuity = 0.0
        for _, row in schedule.iloc[:-1].iterrows():
            accrual_start = row.AccrualStart
            accrual_end   = row.AccrualEnd
            tau           = day_count_fraction(accrual_start, accrual_end, self.float_conv)
            df            = self.__get_df(accrual_end)
            annuity       += tau * df

        # Last period
        last          = schedule.iloc[-1]
        accrual_start = last.AccrualStart
        accrual_end   = last.AccrualEnd
        tau_last      = day_count_fraction(accrual_start, accrual_end, self.float_conv)
        
        # Discount factor
        df = (1 - rate * annuity) / (1 + rate * tau_last)
        return df * self.df_settle
    
    def bootstrap_irs_swap(self, instrument: Instrument):
        fixed_schedule = instrument.fixed_schedule
        float_schedule = instrument.float_schedule
        rate           = instrument.rate

        # Fixed leg annuity – 30U/360 – Discount = ESTR
        fixed_annuity = 0.0
        for _, row in fixed_schedule.iloc[:-1].iterrows():
            tau            = day_count_fraction(row.AccrualStart, row.AccrualEnd, self.fixed_conv)
            df             = self.discount_curve.discount(row.PaymentDate)
            fixed_annuity += tau * df

        # Last fixed period
        last_fixed     = fixed_schedule.iloc[-1]
        tau_last_fixed = day_count_fraction(last_fixed.AccrualStart, last_fixed.AccrualEnd, self.fixed_conv)
        df_last_fixed  = self.discount_curve.discount(last_fixed.PaymentDate)

        # Float leg PV – ACT/360 – Discount = ESTR
        float_pv = 0.0
        for _, row in float_schedule.iloc[:-1].iterrows():
            tau       = day_count_fraction(row.AccrualStart, row.AccrualEnd, self.float_conv)
            df_start  = self.discount_curve.discount(row.AccrualStart)
            df_end    = self.discount_curve.discount(row.AccrualEnd)
            fwd       = (df_start / df_end - 1) / tau
            df_pay    = self.discount_curve.discount(row.PaymentDate)
            float_pv += tau * fwd * df_pay

        # Last float period
        last_float = float_schedule.iloc[-1]
        tau_last_float = day_count_fraction(last_float.AccrualStart, last_float.AccrualEnd, self.float_conv)
        df_start_last = self.discount_curve.discount(last_float.AccrualStart)
        df_pay_last = self.discount_curve.discount(last_float.PaymentDate)

        def equation(x):
            fwd_last = (df_start_last / x - 1) / tau_last_float
            float_last = tau_last_float * fwd_last * df_pay_last
            pv_fixed = rate * (fixed_annuity + tau_last_fixed * df_last_fixed)
            pv_float = float_pv + float_last
            return pv_fixed - pv_float

        return brentq(equation, 1e-6, 2.0)
    
    def __get_df(self, date: Date) -> float:
        date         = np.datetime64(pd.to_datetime(date))

        if self.__interp_cache is None:
            x            = np.array([(i[0] - self.trade_date).days for i in self.z_curve])
            y            = np.log(np.array(list(self.z_curve.values())))
            self.__interp_cache = interp1d(x, y, kind="linear", fill_value="extrapolate")

        target       = (date - self.trade_date).days
        return np.exp(self.__interp_cache(target))
    
    def build(self) -> ZeroCurve:
        maturities = np.array([i[0] for i in self.z_curve])
        dfs        = np.array(list(self.z_curve.values()))
        return ZeroCurve(self.trade_date, maturities, dfs, self.par_curve.curve)
    
    def output(self) -> Df:
        maturities = pd.Series([i[0] for i in self.z_curve], name="Maturity")
        tenors     = pd.Series([i[1] for i in self.z_curve], name="Tenor")
        dfs        = pd.Series(list(self.z_curve.values()), name="DiscountFactor")
        
        df = pd.concat((maturities, tenors, dfs), axis=1)

        df["Curve"]     = self.par_curve.curve
        df["TradeDate"] = self.trade_date

        df.Curve = df.Curve.astype("category")
        df.Tenor = df.Tenor.astype("category")

        df = df[["TradeDate", "Curve", "Tenor", "Maturity", "DiscountFactor"]]
        return df
    
    def __repr__(self):
        return f"Bootstrapper({str(self.trade_date.date())}, par_curve={self.par_curve.curve}{'' if self.discount_curve is None else ', discount_curve=' + self.discount_curve.curve})"

def bootstrap_estr_loop():
    base   = pd.read_parquet("clean_data/zero_rates.parquet")
    estr   = base[base.Curve == "ESTR"]

    dates  = rates.reset_index()
    dates  = sorted(dates[dates.Date > estr.TradeDate.max()].Date.unique())

    new = []

    for date in dates[:]:
        try:
            new.append(Bootstrapper(ParCurve(date, "ESTR")).output())
        except:
            continue

    if len(new) > 0:
        new = pd.concat([base] + new).sort_values(by=["TradeDate", "Curve", "Maturity"]).reset_index(drop=True)
        new.to_parquet("clean_data/zero_rates.parquet", index=False)
        print("Updated correctly.")

    else:
        print("No new data.")

def bootstrap_euribor_loop():
    base    = pd.read_parquet("clean_data/zero_rates.parquet")
    euribor = base[base.Curve == "EURIBOR6M"]

    dates   = rates.reset_index()
    dates   = sorted(dates[dates.Date > euribor.TradeDate.max()].Date.unique())

    new = []

    for date in dates[:]:
        try:
            disc = Bootstrapper(ParCurve(date, "ESTR")).build()
            new.append(Bootstrapper(ParCurve(date, "EURIBOR6M"), disc).output())
            print(f"{date.date()}: OK")
        except:
            continue

    if len(new) > 0:
        new = pd.concat([base] + new).sort_values(by=["TradeDate", "Curve", "Maturity"]).reset_index(drop=True)
        new.to_parquet("clean_data/zero_rates.parquet", index=False)
        print("Updated correctly.")

    else:
        print("No new data.")

def zero_curve_load(trade_date, curve):
    trade_date = np.datetime64(pd.to_datetime(trade_date))
    z_curve    = zero_rates.xs((trade_date, curve))
    maturity   = z_curve.Maturity.tolist()
    tenors     = z_curve.index.tolist()
    dfs        = z_curve.DiscountFactor.tolist()
    conv       = CURVE_CONVENTIONS[curve]["float_leg"]["day_count"]
    return ZeroCurve(trade_date, tenors, maturity, dfs, curve, conv)


