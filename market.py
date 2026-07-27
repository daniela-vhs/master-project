from dates import clean_date, clean_tenor, Tenor, BusinessDay
from curves import ZeroCurve, buscal, schedule_generation, Bootstrapper, ParCurve
from typing import TypeAlias
from hull_white_calibration import HullWhite
from caplet_stripping import caplet_stripping

import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator, interp1d

Date: TypeAlias = np.datetime64

vol_data     = pd.read_parquet("clean_data/vols.parquet").set_index(["Date", "Tenor", "IsATM", "Strike"])
caplets_data = pd.read_parquet("clean_data/cap_stripping.parquet").set_index(["TradeDate", "Tenor", "IsATM", "Strike"])
rates_data   = pd.read_parquet("clean_data/rates.parquet").set_index(["Date", "Curve", "Tenor"])

class CapVolSurface:
    def __init__(self,
        trade_date,
        strikes,
        tenors,
        maturities,
        vols,
        ):
        self.trade_date = clean_date(trade_date)
        self.strikes = strikes
        self.tenors = tenors
        self.maturities = maturities
        self.vols = vols

        self.t_mat = (self.maturities - self.trade_date).astype(int) / 365

        total_var = self.vols ** 2 * self.t_mat[:, None]
        self.__tenor_interp = [PchipInterpolator(self.t_mat, i) for i in total_var.T]

        self.atm_strikes = dict(sorted({tenor: rate / 100 for tenor, rate in rates_data.xs((self.trade_date, "EURIBOR6M"), level=("Date", "Curve")).Rate.to_dict().items() if Tenor(tenor) >= Tenor("3Y") and Tenor(tenor) <= Tenor("10Y")}.items(), key=lambda x: Tenor(x[0])))

    @classmethod
    def from_date(cls, trade_date):
        trade_date  = clean_date(trade_date)
        df          = vol_data.xs((trade_date, False), level=("Date", "IsATM"))
        df          = df.drop([0.125, 0.25], level="Strike").reindex([f"{i + 3}Y" for i in range(8)], level="Tenor")
        strikes     = np.sort(df.index.get_level_values("Strike").unique() / 100)
        tenors      = np.array(sorted([i for i in df.index.get_level_values("Tenor").unique() if Tenor(i) >= Tenor("3Y") and Tenor(i) <= Tenor("10Y")], key=lambda x: Tenor(x)))
        master      = schedule_generation(trade_date, tenors[-1], "Annual", pay_delay="0 Business Days", include_tenor=True).set_index("Tenor")
        maturities  = np.array([
            np.datetime64(pd.Timestamp(master.loc[t].AccrualEnd).date()) if t in master.index
            else np.datetime64(BusinessDay(trade_date, calendar=buscal).shift("2D", "following").shift(t, "modifiedfollowing").date)
            for t in tenors
        ])
        vol_surface = df.reset_index().pivot(index="Tenor", columns="Strike", values="Vol").reindex(tenors).to_numpy()
        
        return cls(trade_date, strikes, tenors, maturities, vol_surface)
    
    def cap_vol(self, maturity, strike):
        maturity = clean_date(maturity)
        target = (maturity - self.trade_date).astype(int) / 365
        clamped_target = np.clip(target, self.t_mat.min(), self.t_mat.max())

        # Date interpolation
        smile = [np.sqrt(f(clamped_target) / clamped_target) for f in self.__tenor_interp]

        return PchipInterpolator(self.strikes, smile)(strike)

    def bump(self, tenor, strike, bump=1):
        tenor = clean_tenor(tenor)
        tenor_idx = np.where(self.tenors == tenor.tenor)[0][0]
        strike_idx = np.where(self.strikes == strike)[0][0]
        vols = np.array(self.vols)
        vols[tenor_idx][strike_idx] += bump
        return CapVolSurface(self.trade_date, self.strikes, self.tenors, self.maturities, vols)
    
    def output(self, tenors=True):
        return pd.DataFrame(self.vols, index=self.tenors if tenors else self.maturities, columns=self.strikes)
    
    def __repr__(self):
        return f"CapVolSurface({self.trade_date})"

    def __getitem__(self, key):
        return self.cap_vol(key[0], key[1])

class CapletVolSurface:
    def __init__(self,
        trade_date,
        strikes,
        tenors,
        fixings,
        vols,
        ):
        self.trade_date = clean_date(trade_date)
        self.strikes = strikes
        self.tenors = tenors
        self.fixings = fixings
        self.vols = vols

        t = (self.fixings - np.datetime64(self.trade_date)).astype("timedelta64[D]").astype(int) / 365
        total_var = self.vols ** 2 * t[:, None]
        self.tfix = t
        self.__tenor_interp = [interp1d(t, i, bounds_error=False, fill_value=(i[0], i[-1])) for i in total_var.T]

    @classmethod
    def from_date(cls, trade_date):
        trade_date  = clean_date(trade_date)
        df          = caplets_data.xs((trade_date, False), level=("TradeDate", "IsATM"))
        strikes     = np.sort(df.index.get_level_values("Strike").unique())
        tenors      = np.array(sorted(df.index.get_level_values("Tenor").unique(), key=lambda x: Tenor(x)))
        fixings     = schedule_generation(trade_date, "10Y", "SemiAnnual", include_tenor=True).set_index("Tenor")
        fixings     = np.array([fixings.loc[i].FixingDate for i in tenors]).astype(np.datetime64)
        vol_surface = df.reset_index().pivot(index="Tenor", columns="Strike", values="StrippedVol").reindex(tenors).to_numpy()
        return cls(trade_date, strikes, tenors, fixings, vol_surface)
    
    def caplet_vol(self, fixing_dates, strike):
        fixing_dates = np.atleast_1d(fixing_dates)
        targets      = np.array([(i - self.trade_date).astype(int) for i in fixing_dates]) / 365
        targets      = np.clip(targets, self.tfix.min(), self.tfix.max())
        strike       = np.clip(strike, self.strikes.min(), self.strikes.max())

        # Smile matrix
        smile_matrix = np.column_stack([np.sqrt(f(targets) / targets) for f in self.__tenor_interp])

        value = PchipInterpolator(self.strikes, smile_matrix, axis=1)(strike)
        return value[0] if len(value) == 1 else value
    
    def bump(self, tenor, bump=1):
        tenor = clean_tenor(tenor)
        idx = np.where(self.tenors == tenor.tenor)[0][0]
        vols = np.array(self.vols)
        vols[idx] += bump
        return CapletVolSurface(self.trade_date, self.strikes, self.tenors, self.fixings, vols)
    
    def output(self):
        return pd.DataFrame(self.vols, columns=self.strikes, index=self.tenors)
    
    def __repr__(self):
        return f"CapletVolSurface({self.trade_date})"

class Market:
    def __init__(self,
            trade_date: Date,
            estr_curve: ZeroCurve,
            euribor_curve: ZeroCurve,
            caplet_surface: CapletVolSurface = None,
            cap_surface: CapVolSurface = None,
            hull_white: HullWhite = None,
            ):
        
        self.trade_date = clean_date(trade_date)
        self.estr_curve = estr_curve
        self.euribor_curve = euribor_curve
        self.caplet_surface = caplet_surface
        self.cap_surface = cap_surface
        self.hull_white = hull_white

    @classmethod
    def from_date(cls, trade_date, generate_caplets=True, generate_caps=True, generate_hw=True):
        trade_date  = clean_date(trade_date)
        z_estr      = ZeroCurve.from_date(trade_date, "ESTR")
        z_eur       = ZeroCurve.from_date(trade_date, "EURIBOR6M")
        caplet_vols = CapletVolSurface.from_date(trade_date) if generate_caplets else None
        cap_vols    = CapVolSurface.from_date(trade_date) if generate_caps else None
        hull_white  = HullWhite.from_date(trade_date) if generate_hw else None
        return cls(trade_date, z_estr, z_eur, caplet_vols, cap_vols, hull_white)
    
    def day_shift(self, other):
        return (self.trade_date - other.trade_date).astype("int")
    
    def __repr__(self):
        return f"Market({str(self.trade_date)})"

    # ----- #
    # BUMPS #
    # ----- #
    # 1. ESTR Curve
    def estr_bump(self, tenor, bump=1):
        curve  = self.estr_curve
        tenors = curve.tenors
        mats   = curve.maturities
        dfs    = np.array(curve.dfs)
        idx    = np.where(tenors == tenor)
        rate   = curve.continuous_rate(mats[idx])
        tau    = (mats[idx][0] - self.trade_date).days / 360

        # Compute bump
        new_rate = rate + bump / 10_000
        new_df   = np.exp(-new_rate * tau)

        # Apply bump
        dfs[idx] = new_df

        # Create new bumped Zero Curve
        new_curve = ZeroCurve(self.trade_date, tenors, mats, dfs, "ESTR", "ACT/360")

        # Return new market object
        return Market(self.trade_date, new_curve, self.euribor_curve, self.caplet_surface, self.cap_surface, self.hull_white)

    # 1. ESTR curve bump
    def estr_bump(self, tenor, bump=1):
        par_curve = ParCurve(self.trade_date, "ESTR")
        bumped   = par_curve.instruments[tenor]

        bumped.rate += bump / 10_000
        par_curve.instruments[tenor] = bumped
        zero_curve = Bootstrapper(par_curve).build()
        return Market(
            self.trade_date,
            zero_curve,
            self.euribor_curve,
            self.caplet_surface,
            self.cap_surface,
            self.hull_white
            )

    # 2. EURIBOR curve bump
    def euribor_bump(self, tenor, bump=1):
        par_curve = ParCurve(self.trade_date, "EURIBOR6M")
        bumped   = par_curve.instruments[tenor]

        bumped.rate += bump / 10_000
        par_curve.instruments[tenor] = bumped
        zero_curve = Bootstrapper(par_curve, self.estr_curve).build()
        return Market(
            self.trade_date,
            self.estr_curve,
            zero_curve,
            self.caplet_surface,
            self.cap_surface,
            self.hull_white
            )

    # 3. Cap surface bump
    def cap_vol_bump(self, tenor, strike, bump=1, caplet_recalibration = True, hw_recalibration = True):
        new_cap_surface = self.cap_surface.bump(tenor, strike, bump)
        new_mkt = Market(
            self.trade_date,
            self.estr_curve,
            self.euribor_curve,
            self.caplet_surface,
            new_cap_surface,
            self.hull_white
        )

        # Caplet Surface Recalibration
        if caplet_recalibration:
            caplet_vols = np.array(self.caplet_surface.vols)
            new_caplet_vols = np.zeros_like(caplet_vols)

            for n, strike in enumerate([-0.005, -0.0025, -0.00125, 0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]):
                new_caplet_vols[:, n] = [i[0] for i in caplet_stripping(new_mkt, strike).values()]

            new_caplet_surface = CapletVolSurface(self.trade_date, self.caplet_surface.strikes, self.caplet_surface.tenors, self.caplet_surface.fixings, new_caplet_vols)
            new_mkt = Market(
                        self.trade_date,
                        self.estr_curve,
                        self.euribor_curve,
                        new_caplet_surface,
                        new_cap_surface,
                        self.hull_white
                    )

        else: new_caplet_surface = self.caplet_surface

        # Hull White Recalibration
        if hw_recalibration:
            new_hull_white = self.hull_white.calibrate(new_mkt)
        else:
            new_hull_white = self.hull_white
            
        return Market(self.trade_date, self.estr_curve, self.euribor_curve, new_caplet_surface, new_cap_surface, new_hull_white)

    # 4. Day bump
    def day_bump(self, bump=1):
        dates = vol_data.index.get_level_values("Date").unique()
        date_idx = np.where(self.trade_date == dates)[0][0]
        next_date = dates[date_idx + bump]
        return Market(next_date, self.estr_curve, self.euribor_curve, self.caplet_surface, self.cap_surface, self.hull_white)

    # ------------------ #
    # REAL MARKET SHIFTS #
    # ------------------ #
    # 1. ESTR curve shift
    def estr_shift(self, other):
        self_estr_par = ParCurve(self.trade_date, "ESTR")
        other_estr_par = ParCurve(other.trade_date, "ESTR")
        return {k: v.rate - other_estr_par.instruments[k].rate for k, v in self_estr_par.instruments.items()}

    # 2. EURIBOR6M curve shift
    def euribor_shift(self, other):
        self_euribor_par = ParCurve(self.trade_date, "EURIBOR6M")
        other_euribor_par = ParCurve(other.trade_date, "EURIBOR6M")
        return {k: v.rate - other_euribor_par.instruments[k].rate for k, v in self_euribor_par.instruments.items()}

    # 3. Cap surface shift
    def cap_vol_shift(self, other):
        vol_dif = self.cap_surface.vols - other.cap_surface.vols

        shifts = dict()
        
        for j, strike in enumerate(self.cap_surface.strikes):
            shifts[strike] = dict()
            for i, tenor in enumerate(self.cap_surface.tenors):
                shifts[strike][str(tenor)] = vol_dif[i][j]
                
        return shifts
    
    # 4. Day shift
    def day_shift(self, other):
        return (other.trade_date - self.trade_date).astype(int)

    # 5. All shifts
    def mkt_shifts(self, other):
        shifts = dict()
        shifts["ESTR"] = self.estr_shift(other)
        shifts["EURIBOR6M"] = self.euribor_shift(other)
        shifts["CapVolSurface"] = self.cap_vol_shift(other)
        shifts["Time"] = self.day_shift(other)
        return shifts

    # 3. Cap surface bump
    def cap_vol_bump(self, tenor, strike, bump=1, caplet_recalibration = True, hw_recalibration = True):
        new_cap_surface = self.cap_surface.bump(tenor, strike, bump)
        new_mkt = Market(
            self.trade_date,
            self.estr_curve,
            self.euribor_curve,
            self.caplet_surface,
            new_cap_surface,
            self.hull_white
        )

        # Caplet Surface Recalibration
        if caplet_recalibration:
            caplet_vols = np.array(self.caplet_surface.vols)
            strike_idy  = np.where(self.caplet_surface.strikes == strike)[0][0]
            new_caplet_vols = np.array([i[0] for i in caplet_stripping(new_mkt, strike).values()])

            caplet_vols[:, strike_idy] = new_caplet_vols

            new_caplet_surface = CapletVolSurface(self.trade_date, self.caplet_surface.strikes, self.caplet_surface.tenors, self.caplet_surface.fixings, caplet_vols)
            new_mkt = Market(
                        self.trade_date,
                        self.estr_curve,
                        self.euribor_curve,
                        new_caplet_surface,
                        new_cap_surface,
                        self.hull_white
                    )

        else: new_caplet_surface = self.caplet_surface

        # Hull White Recalibration
        if hw_recalibration:
            new_hull_white = self.hull_white.calibrate(new_mkt)
        else:
            new_hull_white = self.hull_white
            
        return Market(self.trade_date, self.estr_curve, self.euribor_curve, new_caplet_surface, new_cap_surface, new_hull_white)

    def rebuild(self):
        return Market(self.trade_date, self.estr_curve, self.euribor_curve, self.caplet_surface, self.cap_surface, self.hull_white)

    def generate_bumps(self, max_maturity, min_strike, max_strike, rate_bp=1, vol_bp=1, days=1, strike_support=1):
        bumps = dict()
        market = self.rebuild()

        # Build supports
        # Tenors
        max_maturity = clean_date(max_maturity)
        estr_idx = np.where(pd.to_datetime(max_maturity) <= self.estr_curve.maturities)[0].min()
        estr_tenors = market.estr_curve.tenors[1 : estr_idx + 2]

        euribor_idx = np.where(pd.to_datetime(max_maturity) <= self.euribor_curve.maturities)[0].min()
        euribor_tenors = market.euribor_curve.tenors[1 : euribor_idx + 2]

        vol_tenors_idx = np.where(pd.to_datetime(max_maturity) <= self.cap_surface.maturities)[0].min()
        vol_tenors = self.cap_surface.tenors[ : vol_tenors_idx + 2]

        # Strikes
        min_strike_idx = np.where(self.cap_surface.strikes <= min_strike)[0].max()
        max_strike_idx = np.where(self.cap_surface.strikes >= max_strike)[0].min()
        all_strikes = self.caplet_surface.strikes
        strikes_support = all_strikes[np.maximum(0, min_strike_idx - 2) : np.minimum(max_strike_idx + 3, len(all_strikes))]

        # Estr bumps
        estr_bump = dict()
        for tenor in estr_tenors:
            estr_bump[tenor]         = dict()
            estr_bump[tenor]["up"]   = market.estr_bump(tenor, +rate_bp)
            estr_bump[tenor]["mid"]  = market
            estr_bump[tenor]["down"] = market.estr_bump(tenor, -rate_bp)
            estr_bump[tenor]["bp"]   = rate_bp

        bumps["ESTR"] = estr_bump

        # Euribor bumps
        eur_bump = dict()
        for tenor in euribor_tenors:
            eur_bump[tenor]         = dict()
            eur_bump[tenor]["up"]   = market.euribor_bump(tenor, +rate_bp)
            eur_bump[tenor]["mid"]  = market
            eur_bump[tenor]["down"] = market.euribor_bump(tenor, -rate_bp)
            eur_bump[tenor]["bp"]   = rate_bp

        bumps["EURIBOR6M"] = eur_bump

        # Vol bumps
        # Decide boundary strikes for Hull-White regeneration
        atm_strikes = np.array(list(self.cap_surface.atm_strikes.values()))

        day_min_strike = all_strikes[np.where(all_strikes <= atm_strikes.min())[0].max()]
        day_max_strike = all_strikes[np.where(all_strikes >= atm_strikes.max())[0].min()]
        day_min_strike_idx = np.where(all_strikes == day_min_strike)[0][0]
        day_max_strike_idx = np.where(all_strikes == day_max_strike)[0][0]
        strikes_hw = all_strikes[np.maximum(day_min_strike_idx - strike_support + 1, 0) : np.minimum(day_max_strike_idx + strike_support, len(all_strikes))]

        vol_bump = dict()
        for strike in strikes_support:
            vol_bump[strike] = dict()
            for tenor in vol_tenors:
                # Is hull white necessary?
                if strike in strikes_hw:
                    hw_calibration = True
                else:
                    hw_calibration = False

                # Bumps
                vol_bump[strike][tenor]         = dict()
                vol_bump[strike][tenor]["up"]   = market.cap_vol_bump(tenor, strike, +vol_bp, hw_recalibration=hw_calibration)
                vol_bump[strike][tenor]["mid"]  = market
                vol_bump[strike][tenor]["down"] = market.cap_vol_bump(tenor, strike, -vol_bp, hw_recalibration=hw_calibration)
                vol_bump[strike][tenor]["bp"]   = vol_bp

        bumps["CapVolSurface"] = vol_bump

        bumps["Time"] = dict()
        bumps["Time"]["up"] = self.day_bump(days)
        bumps["Time"]["mid"] = market

        return bumps


