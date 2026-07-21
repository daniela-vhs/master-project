import numpy as np
import pandas as pd

from dates import Tenor, Date, day_count_fraction, clean_date, clean_tenor
from curves import buscal, schedule_generation, ZeroCurve, get_fixing, ois_reset_dates, CURVE_CONVENTIONS
from scipy.stats import norm

rates         = pd.read_parquet("clean_data/rates.parquet").set_index(["Date", "Curve", "Tenor"])
vols          = pd.read_parquet("clean_data/vols.parquet").set_index(["Date", "Tenor", "IsATM", "Strike"])
caplets       = pd.read_parquet("clean_data/caplets.parquet").set_index(["TradeDate", "Tenor"])
cap_stripping = pd.read_parquet("clean_data/cap_stripping.parquet").set_index(["TradeDate", "Tenor", "IsATM", "Strike"]).sort_index()

class Instrument:
    def value(self, market, *params, ul):
        return np.zeros_like(ul)

    def delta(self, market, *params, ul):
        return np.zeros_like(ul)
    
    def theta(self, market, *params, ul):
        return np.zeros_like(ul)
    
    def gamma(self, market, *params, ul):
        return np.zeros_like(ul)
    
    def vega(self, market, *params, ul):
        return np.zeros_like(ul)
    
    def volga(self, market, *params, ul):
        return np.zeros_like(ul)
    
    def vanna(self, market, *params, ul):
        return np.zeros_like(ul)

# -- IRS Pricing -------------------------------------------
class IRS(Instrument):
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
    
    def delta(self, bump_fn, tenor, eps_bp=1, ul=None):
        up = bump_fn(tenor, +eps_bp, generate_caplets=False)
        down = bump_fn(tenor, -eps_bp, generate_caplets=False)

        p_up = self.value(up, ul)
        p_down = self.value(down, ul)
        return (p_up - p_down) / (2 * eps_bp / 10_000)
    
    def theta(self, market, eps_days=1, ul=None):
        up = market.bump_date(eps_days)

        p_up = self.value(up, ul)
        return p_up
    
    def __repr__(self):
        return f"IRS({self.float_index}, {self.tenor.tenor}, {self.fixed_rate:.4%})"

class Caplet(Instrument):
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
    def from_date_strike(cls, trade_date, strike):
        trade_date = clean_date(trade_date)
        df = caplets.loc[trade_date].copy()
        caplet_list = dict()

        for tenor, row in df.iterrows():            
            bucket = row.CapTenorBucket if Tenor(tenor) <= Tenor("3Y") else tenor
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
    
    def __repr__(self):
        return f"Caplet({self.tenor.tenor}, Vol: {self.caplet_vol:.4f}, Fwd Rate: {self.fwd_rate:.4%})"
    
    def value(self, market, strike):
        vol         = market.v_surface.caplet_vol(self.fixing_date, strike)
        tfix        = day_count_fraction(market.trade_date, self.fixing_date, "ACT/365")
        fwd_rate    = market.euribor_curve.forward_rate(self.accrual_start, self.accrual_end)
        d           = (fwd_rate - strike) / (vol / 10_000 * np.sqrt(tfix))
        estr_pay_df = market.estr_curve.discount(self.payment_date)
        price = self.tau * estr_pay_df * ((fwd_rate - strike) * norm.cdf(d) + vol / 10_000 * np.sqrt(tfix) * norm.pdf(d))
        return price
    
    def delta(self, bump_fn, tenor, eps_bp=1, ul=None):
        up = bump_fn(tenor, +eps_bp, generate_caplets=False)
        down = bump_fn(tenor, -eps_bp, generate_caplets=False)

        p_up = self.value(up, ul)
        p_down = self.value(down, ul)
        return (p_up - p_down) / (2 * eps_bp / 10_000)
    
    def theta(self, market, eps_days=1, ul=None):
        up = market.bump_date(eps_days)

        p_up = self.value(up, ul)
        return p_up

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
    
    def __repr__(self):
        return f"Cap({self.tenor.tenor}, Vol: {self.cap_vol:.4f}, Strike: {self.strike:.4%})"
    
    @classmethod
    def from_date_strike(cls, trade_date, market, strike=None):
        trade_date = clean_date(trade_date)
        atm = strike is None

        if atm:
            df = vols.xs((trade_date, True), level=("Date", "IsATM")).droplevel("Strike")
            template_caplets = Caplet.from_date_strike(trade_date, 0.0)  # schedule/DF template only, vol discarded below
        else:
            df = vols.xs((trade_date, strike * 100), level=("Date", "Strike")).droplevel("IsATM")
            all_caplets = Caplet.from_date_strike(trade_date, strike)

        anchors = [i for i in df.index if Tenor(i) >= Tenor("3Y") and Tenor(i) <= Tenor("10Y")]
        df = df.loc[anchors]

        caps = dict()
        for tenor, row in df.iterrows():
            if atm:
                cap_strike = IRS.from_benchmark("EURIBOR6M", trade_date, tenor).par_rate(market)
                caplets = {}
                for t, c in template_caplets.items():
                    if c.isin(tenor):
                        vol = market.caplet_surface.caplet_vol(c.fixing_date, cap_strike)
                        caplets[t] = Caplet(c.trade_date, c.tenor, c.cap_tenor_bucket, c.fixing_date,
                                            c.accrual_start, c.accrual_end, c.payment_date, c.tau, c.tfix,
                                            c.eur_start_df, c.eur_end_df, c.estr_pay_df, c.fwd_rate, vol)
            else:
                cap_strike = strike
                caplets = {t: c for t, c in all_caplets.items() if c.isin(tenor)}

            caps[tenor] = Cap(trade_date, tenor, row.Vol, cap_strike, caplets)

        return dict(sorted(caps.items(), key=lambda x: x[1].tenor))
    
    @classmethod
    def from_date(cls, trade_date, market):
        strikes = [None, -0.005, -0.0025, -0.00125, 0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]
        caplet_list = dict()

        for strike in strikes:
            key = "ATM" if strike is None else strike
            caplet_list[key] = cls.from_date_strike(trade_date, market, strike)

        return caplet_list
    
    @classmethod
    def from_market(cls, market, strike=None):
        atm = strike is None

        if atm:
            strikes = cap_stripping.sort_index().xs((market.trade_date, atm), level=("TradeDate", "IsATM")).reset_index(level="Strike")
            flat_vols = vols.xs((market.trade_date, atm), level=("Date", "IsATM")).droplevel("Strike").Vol
        else:
            strikes = cap_stripping.sort_index().xs((market.trade_date, strike), level=("TradeDate", "Strike")).droplevel("IsATM")
            strikes["Strike"] = strike
            flat_vols = vols.xs((market.trade_date, strike * 100), level=("Date", "Strike")).droplevel("IsATM").Vol
        
        cap_list  = {}

        for tenor, row in strikes.iterrows():
            caplets = dict([(k, v) for k, v in market.caplets.items() if v.isin(tenor)])
            cap_list[tenor] = Cap(market.trade_date, tenor, flat_vols.loc[tenor], row.Strike, caplets)

        return cap_list

def caplet_load(trade_date: Date, strike=None):
    trade_date = clean_date(trade_date)
    df = caplets.loc[trade_date].copy()
    caplet_list = dict()
    atm = True if strike is None else False

    for tenor, row in df.iterrows():
        bucket = row.CapTenorBucket
        if atm:
            stripped_vol = cap_stripping.loc[(trade_date, bucket, True)].iloc[0].StrippedVol
        else:
            stripped_vol = cap_stripping.loc[(trade_date, bucket, atm, strike)].StrippedVol
        caplet_list[tenor] = Caplet(trade_date, tenor, bucket, row.FixingDate, row.AccrualStart, row.AccrualEnd, row.PaymentDate, row.Tau, row.TFix, row.EurStartDF, row.EurEndDF, row.EstrPayDF, row.FwdRate, stripped_vol)
    
    return caplet_list

def cap_load(trade_date: Date, strike=None):
    trade_date = clean_date(trade_date)
    all_caplets = caplet_load(trade_date, strike)
    atm = strike is None

    if atm:
        df = vols.xs((trade_date, True), level=("Date", "IsATM")).copy().droplevel("Strike")
    else:
        df = vols.xs((trade_date, strike * 100), level=("Date", "Strike")).copy().droplevel("IsATM")
    
    anchors = [i for i in df.index if Tenor(i) >= Tenor("3Y") and Tenor(i) <= Tenor("10Y")]
    df = df.loc[anchors]

    if atm:
        df = df.join(cap_stripping.xs((trade_date, True), level=("TradeDate", "IsATM")).reset_index().set_index("Tenor"))
    else:
        df["Strike"] = strike

    caps = dict()
    for tenor, row in df.iterrows():
        caplets = dict([(k, v) for k, v in all_caplets.items() if v.isin(tenor)])
        caps[tenor] = Cap(trade_date, tenor, row.Vol, row.Strike, caplets)

    caps = dict(sorted(caps.items(), key=lambda x: x[1].tenor))

    return caps

def caps_from_market(market, strike=None):
    atm = strike is None

    if atm:
        strikes = cap_stripping.sort_index().xs((market.trade_date, atm), level=("TradeDate", "IsATM")).reset_index(level="Strike")
        flat_vols = vols.xs((market.trade_date, atm), level=("Date", "IsATM")).droplevel("Strike").Vol
    else:
        strikes = cap_stripping.sort_index().xs((market.trade_date, strike), level=("TradeDate", "Strike")).droplevel("IsATM")
        strikes["Strike"] = strike
        flat_vols = vols.xs((market.trade_date, strike * 100), level=("Date", "Strike")).droplevel("IsATM").Vol
    
    cap_list  = {}

    for tenor, row in strikes.iterrows():
        caplets = dict([(k, v) for k, v in market.caplets.items() if v.isin(tenor)])
        cap_list[tenor] = Cap(market.trade_date, tenor, flat_vols.loc[tenor], row.Strike, caplets)

    return cap_list


