import numpy as np
import pandas as pd

from dates import Tenor, Date, day_count_fraction, clean_date, clean_tenor
from curves import buscal, schedule_generation, get_fixing, ois_reset_dates, CURVE_CONVENTIONS
from scipy.stats import norm

rates         = pd.read_parquet("clean_data/rates.parquet").set_index(["Date", "Curve", "Tenor"])
vols          = pd.read_parquet("clean_data/vols.parquet").set_index(["Date", "Tenor", "IsATM", "Strike"])
caplets       = pd.read_parquet("clean_data/caplets.parquet").set_index(["TradeDate", "Tenor"])
cap_stripping = pd.read_parquet("clean_data/cap_stripping.parquet").set_index(["TradeDate", "Tenor", "IsATM", "Strike"]).sort_index()

# -- IRS Pricing -------------------------------------------
class IRS:
    def __init__(self,
        trade_date : Date,
        tenor      : Tenor,
        settlement : str,

        # Fixed Leg
        fixed_leg_conv      : str,
        fixed_leg_freq      : str,
        fixed_leg_pay_delay : str,

        # Float Leg
        float_leg_conv      : str,
        float_leg_freq      : str,
        float_leg_pay_delay : str,
        float_leg_fix_lag   : str,

        fixed_rate  : float,
        float_index : str,
        cal         : np.busdaycalendar=buscal): # True: pay fix, False: rcv fix
        
        self.trade_date  = trade_date
        self.fixed_rate  = fixed_rate
        self.float_index = float_index
        self.is_ois      = float_index == "ESTR"
        self.cal         = cal
        self.settlement  = settlement
        self.tenor       = Tenor(tenor) if isinstance(tenor, str) else tenor

        self.fixed_leg_conv      = fixed_leg_conv
        self.fixed_leg_freq      = fixed_leg_freq
        self.fixed_leg_pay_delay = fixed_leg_pay_delay

        self.float_leg_conv      = float_leg_conv
        self.float_leg_freq      = float_leg_freq
        self.float_leg_pay_delay = float_leg_pay_delay
        self.float_leg_fix_lag   = float_leg_fix_lag

        # Fixed schedule
        self.fixed_schedule = schedule_generation(
            trade_date=self.trade_date, tenor=self.tenor, frequency=self.fixed_leg_freq,
            fixing_lag="0 Business Days", pay_delay=self.fixed_leg_pay_delay, cal=self.cal
        )

        # Float schedule
        self.float_schedule = schedule_generation(
            trade_date=self.trade_date, tenor=self.tenor, frequency=self.float_leg_freq,
            fixing_lag=self.float_leg_fix_lag, pay_delay=self.float_leg_pay_delay, cal=self.cal
        )

    @classmethod
    def from_benchmark(cls, curve, trade_date, tenor):
        conv = CURVE_CONVENTIONS[curve]
        trade_date = clean_date(trade_date)
        fix_leg = conv["fixed_leg"]
        flt_leg = conv["float_leg"]
        return cls(trade_date, tenor, conv["settlement"], fix_leg["day_count"], fix_leg["pay_freq"], fix_leg["pay_delay"], flt_leg["day_count"], flt_leg["pay_freq"], flt_leg["pay_delay"], flt_leg["fixing_lag"], 0, curve, buscal)
    
    def fixed_leg(self, market, fixed_rate=None):
        fixed_rate = np.atleast_1d(self.fixed_rate) if fixed_rate is None else np.atleast_1d(fixed_rate)
        df = self.fixed_schedule.copy()
        df = df[df.PaymentDate > market.trade_date]

        tau = day_count_fraction(df.AccrualStart, df.AccrualEnd, self.fixed_leg_conv)
        df  = market.estr_curve.discount(df.PaymentDate.to_numpy())
        return np.sum(tau * df), np.sum(fixed_rate.reshape(-1, 1) @ (tau * df).reshape(1, -1), axis=1)
    
    def term_float_leg(self, market):
        df = self.float_schedule.copy()
        df = df[df.PaymentDate > market.trade_date]

        tau = day_count_fraction(df.AccrualStart, df.AccrualEnd, self.float_leg_conv)

        is_fixed = df.FixingDate.to_numpy() < market.trade_date

        # Fixed periods
        fixed_coupons = df.FixingDate.to_numpy()[is_fixed]
        if len(fixed_coupons) > 0:
            fixed = np.atleast_1d(get_fixing(df.FixingDate.to_numpy()[is_fixed], self.float_index))

        # Unfixed
        unfixed   = ~is_fixed
        fwd_start = market.euribor_curve.discount(df.AccrualStart.to_numpy()[unfixed])
        fwd_end   = market.euribor_curve.discount(df.AccrualEnd.to_numpy()[unfixed])
        fwd_rate  = (fwd_start / fwd_end - 1) / tau[unfixed]
        
        # Discount
        dfs = market.estr_curve.discount(df.PaymentDate.to_numpy())

        if len(fixed_coupons) > 0:
            return np.sum(np.concatenate((fixed, fwd_rate)) * tau * dfs)

        return np.sum(fwd_rate * tau * dfs)
    
    def ois_float_leg(self, market):
        df = self.float_schedule.copy()
        df = df[df.PaymentDate > market.trade_date]

        tau      = day_count_fraction(df.AccrualStart, df.AccrualEnd, self.float_leg_conv)
        fwd_rate = []
        
        for i, row in df.iterrows():
            fwd_rate.append(self.ois_rate(row.AccrualStart, row.AccrualEnd, tau[i], market))

        dfs = market.estr_curve.discount(df.PaymentDate.to_numpy())

        return np.sum(fwd_rate * tau * dfs)
    
    def ois_rate(self, start, end, tau, market) -> float:
        start, end = map(lambda x: np.datetime64(pd.to_datetime(x).date()), (start, end))
        realized_end = np.minimum(end, np.maximum(start, market.trade_date))

        growth = 1.0

        if realized_end > start:
            resets = ois_reset_dates(start, realized_end, self.cal)
            fixings = get_fixing(resets.Date.to_numpy(), self.float_index)
            growth *= np.prod(1 + fixings * resets.Delta.to_numpy())

        if end > realized_end:
            growth *= market.estr_curve.discount(realized_end) / market.estr_curve.discount(end)
        
        return (growth - 1) / tau
    
    def float_leg(self, market):
        if self.is_ois:
            return self.ois_float_leg(market)
        else:
            return self.term_float_leg(market)
    
    def par_rate(self, market):
        float_pv = self.float_leg(market)
        annuity = self.fixed_leg(market)[0]
        
        return (float_pv / annuity)
    
    def value(self, market, ul=None):
        fixed_pv = self.fixed_leg(market, ul)[1]
        float_pv = self.float_leg(market)
        return float_pv - fixed_pv
    
    def __repr__(self):
        return f"IRS({self.float_index}, {self.tenor.tenor}, {self.fixed_rate:.4%})"

class Caplet:
    def __init__(self,
                 trade_date: Date,
                 tenor: Tenor,
                 cap_tenor_bucket: Tenor,
                 fixing_date: Date,
                 accrual_start: Date,
                 accrual_end: Date,
                 payment_date: Date,
                 tau: float,
                 tfix: float,
                 eur_start_df: float,
                 eur_end_df: float,
                 estr_pay_df: float,
                 fwd_rate: float,
                 caplet_vol: float,
                 ):
        
        self.trade_date       = clean_date(trade_date)
        self.tenor            = clean_tenor(tenor)
        self.cap_tenor_bucket = clean_tenor(cap_tenor_bucket)
        self.fixing_date      = clean_date(fixing_date)
        self.accrual_start    = clean_date(accrual_start)
        self.accrual_end      = clean_date(accrual_end)
        self.payment_date     = clean_date(payment_date)
        self.tau              = tau
        self.tfix             = tfix
        self.eur_start_df     = eur_start_df
        self.eur_end_df       = eur_end_df
        self.estr_pay_df      = estr_pay_df
        self.fwd_rate         = fwd_rate
        self.caplet_vol       = caplet_vol
        self.tdelta            = day_count_fraction(self.accrual_start, self.accrual_end, "ACT/365")

    @classmethod
    def generate(cls, market, vol=0):
        tenors = [Tenor("3Y") + Tenor(f"{i * 6}M") for i in range(15)]

        schedule = schedule_generation(market.trade_date, "10Y", "SemiAnnual", pay_delay="0 Business Days", include_tenor=True).set_index("Tenor")
        schedule = schedule[schedule.FixingDate > market.trade_date]

        fixing_date   = schedule.FixingDate.tolist()
        accrual_start = schedule.AccrualStart.tolist()
        accrual_end   = schedule.AccrualEnd.tolist()
        payment_date  = schedule.PaymentDate.tolist()
        tau           = day_count_fraction(accrual_start, accrual_end, "ACT/360")
        tfix          = day_count_fraction(np.repeat(market.trade_date, len(accrual_end)), fixing_date, "ACT/365")
        eur_start_df  = market.euribor_curve[accrual_start]
        eur_end_df    = market.euribor_curve[accrual_end]
        estr_pay_df   = market.estr_curve[payment_date]
        fwd_rate      = (eur_start_df / eur_end_df - 1) / tau

        caplets = dict()

        for n, tenor in enumerate(schedule.index):
            bucket = [i.tenor for i in tenors if Tenor(tenor) <= i][0]
            caplets[tenor] = Caplet(
                market.trade_date, tenor, bucket, fixing_date[n], accrual_start[n], accrual_end[n], payment_date[n],
                tau[n], tfix[n], eur_start_df[n], eur_end_df[n], estr_pay_df[n], fwd_rate[n], vol
            )
        
        return caplets

    @classmethod
    def from_date_strike(cls, trade_date, strike):
        trade_date = clean_date(trade_date)
        df = caplets.loc[trade_date].copy()
        caplet_list = dict()

        for tenor, row in df.iterrows():            
            bucket = row.CapTenorBucket
            stripped_vol = cap_stripping.loc[(trade_date, bucket, False, strike)].StrippedVol

            caplet_list[tenor] = Caplet(trade_date, tenor, bucket, row.FixingDate, row.AccrualStart, row.AccrualEnd, row.PaymentDate, row.Tau, row.TFix, row.EurStartDF, row.EurEndDF, row.EstrPayDF, row.FwdRate, stripped_vol)
        return caplet_list
    
    @classmethod
    def from_date(cls, trade_date):
        strikes = [-0.005, -0.0025, -0.00125, 0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]
        caplet_list = dict()

        for strike in strikes:
            key = "ATM" if strike is None else strike
            caplet_list[key] = cls.from_date_strike(trade_date, strike)

        return caplet_list

    def bachelier_price(self, strike, vol=None):
        vol   = self.caplet_vol if vol is None else vol
        d     = (self.fwd_rate - strike) / (vol / 10_000 * np.sqrt(self.tfix))
        price = self.tau * self.estr_pay_df * ((self.fwd_rate - strike) * norm.cdf(d) + vol / 10_000 * np.sqrt(self.tfix) * norm.pdf(d))
        return price
    
    def jamshidian_price(self, strike, a, sigma):
        tdelta      = self.tdelta
        bond_strike = 1 / (1 + strike * self.tau)
        B_ts        = (1 - np.exp(-a * tdelta)) / a
        sigma_p     = sigma * np.sqrt((1 - np.exp(-2 * a * self.tfix)) / (2 * a)) * B_ts
        h           = 1 / sigma_p * np.log(self.eur_end_df / (self.eur_start_df * bond_strike)) + sigma_p / 2
        zbp         = bond_strike * self.eur_start_df * norm.cdf(-h + sigma_p) - self.eur_end_df * norm.cdf(-h)
        price       = (1 + strike * self.tau) * zbp * self.estr_pay_df / self.eur_end_df
        return price
    
    def isin(self, parent_tenor: Tenor):
        parent_tenor = clean_tenor(parent_tenor)
        return self.tenor <= parent_tenor
    
    def rebuild(self):
        return Caplet(
            self.trade_date,
            self.tenor,
            self.cap_tenor_bucket,
            self.fixing_date,
            self.accrual_start,
            self.accrual_end,
            self.payment_date,
            self.tau,
            self.tfix,
            self.eur_start_df,
            self.eur_end_df,
            self.estr_pay_df,
            self.fwd_rate,
            self.caplet_vol
        )

    def rebuild_rates(self, market):
        eur_start_df  = market.euribor_curve[self.accrual_start]
        eur_end_df    = market.euribor_curve[self.accrual_end]
        estr_pay_df   = market.estr_curve[self.payment_date]
        fwd_rate      = (eur_start_df / eur_end_df - 1) / self.tau
        return Caplet(
            self.trade_date,
            self.tenor,
            self.cap_tenor_bucket,
            self.fixing_date,
            self.accrual_start,
            self.accrual_end,
            self.payment_date,
            self.tau,
            self.tfix,
            eur_start_df,
            eur_end_df,
            estr_pay_df,
            fwd_rate,
            self.caplet_vol
        )

    def rebuild_time(self, market):
        tfix = day_count_fraction(market.trade_date, self.fixing_date, "ACT/365")
        return Caplet(
            self.trade_date,
            self.tenor,
            self.cap_tenor_bucket,
            self.fixing_date,
            self.accrual_start,
            self.accrual_end,
            self.payment_date,
            self.tau,
            tfix,
            self.eur_start_df,
            self.eur_end_df,
            self.estr_pay_df,
            self.fwd_rate,
            self.caplet_vol
        )

    def rebuild_vol(self, market, strike):
        caplet_vol = market.caplet_surface.caplet_vol(self.fixing_date, strike)
        return Caplet(
            self.trade_date,
            self.tenor,
            self.cap_tenor_bucket,
            self.fixing_date,
            self.accrual_start,
            self.accrual_end,
            self.payment_date,
            self.tau,
            self.tfix,
            self.eur_start_df,
            self.eur_end_df,
            self.estr_pay_df,
            self.fwd_rate,
            caplet_vol,
        )
    
    def __repr__(self):
        return f"Caplet({self.tenor.tenor}, Vol: {self.caplet_vol:.4f}, Fwd Rate: {self.fwd_rate:.4%})"

class Cap:
    def __init__(self,
                 trade_date: Date,
                 tenor: Tenor,
                 cap_vol: float,
                 strike: float,
                 caplets: dict,
                 ):
        
        self.trade_date = clean_date(trade_date)
        self.tenor = clean_tenor(tenor)
        self.cap_vol = cap_vol
        self.strike = strike
        self.caplets = caplets

    @classmethod
    def from_market_strike(cls, market, strike=None, flat_vol=False):
        atm = strike is None

        if atm:
            df = vols.xs((market.trade_date, True), level=("Date", "IsATM")).droplevel("Strike")
            all_caplets = {k: v.rebuild() for k, v in Caplet.from_date_strike(market.trade_date, 0.0).items()}

        else:
            df = vols.xs((market.trade_date, strike * 100), level=("Date", "Strike")).droplevel("IsATM")
            all_caplets = {k: v.rebuild() for k, v in Caplet.from_date_strike(market.trade_date, strike).items()}

        anchors = sorted([i for i in df.index if Tenor(i) >= Tenor("3Y") and Tenor(i) <= Tenor("10Y")], key=lambda x: Tenor(x))
        df = df.loc[anchors]

        caps = dict()

        for tenor, row in df.iterrows():
            if atm:
                cap_strike = IRS.from_benchmark("EURIBOR6M", market.trade_date, tenor).par_rate(market)
                caplets = dict()

                for t, c in all_caplets.items():
                    if c.isin(tenor):
                        if flat_vol:
                            vol = row.Vol
                        else:
                            vol = market.caplet_surface.caplet_vol(c.fixing_date, cap_strike)
                        caplets[t] = Caplet(
                            c.trade_date, c.tenor, c.cap_tenor_bucket, c.fixing_date,
                            c.accrual_start, c.accrual_end, c.payment_date, c.tau, c.tfix,
                            c.eur_start_df, c.eur_end_df, c.estr_pay_df, c.fwd_rate, vol
                        )
            else:
                cap_strike = strike
                caplets = {t: c.rebuild() for t, c in all_caplets.items() if c.isin(tenor)}

                if flat_vol:
                    for caplet in caplets.values():
                        caplet.caplet_vol = row.Vol

            caps[tenor] = Cap(market.trade_date, tenor, row.Vol, cap_strike, caplets)

        return caps

    @classmethod
    def from_market(cls, market, flat_vol=False):
        strikes = [None, -0.005, -0.0025, -0.00125, 0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]
        caps = {("ATM" if i is None else i): cls.from_market_strike(market, i, flat_vol) for i in strikes}
        return caps

    @classmethod
    def generate(cls, market, tenor, strike=None, flat_vol=True, caplets=None):
        if strike is None:
            strike = IRS.from_benchmark("EURIBOR6M", market.trade_date, tenor).par_rate(market)

        tenor    = clean_tenor(tenor)
        schedule = schedule_generation(market.trade_date, "10Y", "SemiAnnual", pay_delay="0 Business Days", include_tenor=True).set_index("Tenor")
        schedule = schedule[[Tenor(t) <= tenor for t in schedule.index]]
        schedule = schedule[schedule.FixingDate > market.trade_date]
        cap_vol  = market.cap_surface.cap_vol(schedule.iloc[-1].AccrualEnd, strike)

        if caplets is not None:
            caplets = {k: v.rebuild() for k, v in caplets.items() if v.isin(tenor)}
            for caplet in caplets.values():
                caplet.caplet_vol = cap_vol
            return cls(market.trade_date, tenor, cap_vol, strike, caplets)

        anchors = [Tenor("3Y") + Tenor(f"{i * 6}M") for i in range(15) if Tenor("3Y") + Tenor(f"{i * 6}M") <= tenor]

        # Caplet vols
        if not flat_vol:
            vols = [market.caplet_surface.caplet_vol(i, strike) for i in schedule.FixingDate]
        else:
            vols = np.ones(len(schedule.FixingDate)) * cap_vol

        # Caplet information
        fixing_date   = schedule.FixingDate.tolist()
        accrual_start = schedule.AccrualStart.tolist()
        accrual_end   = schedule.AccrualEnd.tolist()
        payment_date  = schedule.PaymentDate.tolist()
        tau           = day_count_fraction(accrual_start, accrual_end, "ACT/360")
        tfix          = day_count_fraction(np.repeat(market.trade_date, len(accrual_end)), fixing_date, "ACT/365")
        eur_start_df  = market.euribor_curve[accrual_start]
        eur_end_df    = market.euribor_curve[accrual_end]
        estr_pay_df   = market.estr_curve[payment_date]
        fwd_rate      = (eur_start_df / eur_end_df - 1) / tau

        caplet_list = dict()

        for n, t in enumerate(schedule.index):
            bucket = [i.tenor for i in anchors if Tenor(t) <= i][0]
            caplet_list[t] = Caplet(market.trade_date, t, bucket, fixing_date[n], accrual_start[n], accrual_end[n], payment_date[n], tau[n], tfix[n], eur_start_df[n], eur_end_df[n], estr_pay_df[n], fwd_rate[n], vols[n])

        return cls(market.trade_date, tenor, cap_vol, strike, caplet_list)

    def bachelier_price(self, flat_vol=False):
        price = 0

        if len(self.caplets) > 0:
            for caplet in self.caplets.values():
                if flat_vol:
                    price += caplet.bachelier_price(self.strike, self.cap_vol)
                else:
                    price += caplet.bachelier_price(self.strike)

        return price
    
    def jamshidian_price(self, a, sigma):
        price = 0

        if len(self.caplets) > 0:
            for caplet in self.caplets.values():
                price += caplet.jamshidian_price(self.strike, a, sigma)

        return price
    
    def stripping_pricing_error(self):
        return self.bachelier_price(True) - self.bachelier_price()
    
    def hw_pricing_error(self, a, sigma):
        return self.bachelier_price(True) - self.jamshidian_price(a, sigma)

    def rebuild_market(self, market):
            # New cap vol
            maturity = sorted(self.caplets.items(), key=lambda x: Tenor(x[0]))[-1][1].accrual_end
            cap_vol = market.cap_surface.cap_vol(maturity, self.strike)
    
            # New caplet vols and tfix, remove fixed caplets from list
            caplets = {tenor: caplet.rebuild_rates(market) for tenor, caplet in self.caplets.items() if caplet.fixing_date > market.trade_date}
            fixings = [i.fixing_date for i in caplets.values()]
            tfix    = day_count_fraction(np.repeat(market.trade_date, len(fixings)), fixings, "ACT/365")
            caplet_vols = market.caplet_surface.caplet_vol(fixings, self.strike)
    
            for n, caplet in enumerate(caplets.values()):
                caplet.caplet_vol = caplet_vols[n]
                caplet.tfix = tfix[n]
    
            return Cap(self.trade_date, self.tenor, cap_vol, self.strike, caplets)

    def rebuild_vol(self, market):
        # First cap vol
        maturity = self.caplets[self.tenor.tenor].accrual_end
        cap_vol  = market.cap_surface.cap_vol(maturity, self.strike)

        # Caplet vol
        caplets = {tenor: caplet.rebuild() for tenor, caplet in self.caplets.items()}
        new_vols = market.caplet_surface.caplet_vol([i.fixing_date for i in caplets.values()], self.strike)
        for i, caplet in enumerate(caplets.values()):
            caplet.caplet_vol = new_vols[i]

        return Cap(self.trade_date, self.tenor, cap_vol, self.strike, caplets)

    def rebuild_rates(self, market):
        # Only caplets
        caplets = {tenor: caplet.rebuild_rates(market) for tenor, caplet in self.caplets.items()}
        return Cap(self.trade_date, self.tenor, self.cap_vol, self.strike, caplets)

    def rebuild_time(self, market):
        # Only caplets
        caplets = {tenor: caplet.rebuild_time(market) for tenor, caplet in self.caplets.items() if caplet.fixing_date > market.trade_date}
        return Cap(self.trade_date, self.tenor, self.cap_vol, self.strike, caplets)

    def __repr__(self):
        return f"Cap({self.tenor.tenor}, Vol: {self.cap_vol:.4f}, Strike: {self.strike:.4%})"


