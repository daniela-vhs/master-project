from dates import clean_date, clean_tenor, Tenor, BusinessDay
from curves import ZeroCurve, buscal, schedule_generation
from typing import TypeAlias

import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator, interp1d

Date: TypeAlias = np.datetime64

vol_data    = pd.read_parquet("clean_data/vols.parquet").set_index(["Date", "Tenor", "IsATM", "Strike"])
caplets_data = pd.read_parquet("clean_data/cap_stripping.parquet").set_index(["TradeDate", "Tenor", "IsATM", "Strike"])

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

    @classmethod
    def from_date(cls, trade_date):
        trade_date  = clean_date(trade_date)
        df          = vol_data.xs((trade_date, False), level=("Date", "IsATM"))
        strikes     = np.sort(df.index.get_level_values("Strike").unique() / 100)
        tenors      = np.array(sorted(df.index.get_level_values("Tenor").unique(), key=lambda x: Tenor(x)))
        maturities  = [BusinessDay(trade_date, calendar=buscal).shift("2D", "following").shift(i, "modifiedfollowing").date for i in tenors]
        vol_surface = df.reset_index().pivot(index="Tenor", columns="Strike", values="Vol").reindex(tenors).to_numpy()
        
        return cls(trade_date, strikes, tenors, maturities, vol_surface)
    
    def cap_vol(self, maturity, strike):
        maturity = clean_date(maturity)
        target = (maturity - self.trade_date).astype(int) / 365
        clamped_target = np.clip(target, self.t_mat.min(), self.t_mat.max())

        # Date interpolation
        smile = [np.sqrt(f(clamped_target) / clamped_target) for f in self.__tenor_interp]

        return PchipInterpolator(self.strikes, smile)(strike)
    
    def bump(self, tenor, bump=1):
        tenor = clean_tenor(tenor)
        tenor_idx = np.where(self.tenors == tenor.tenor)
        vols = np.array(self.vols)
        vols[tenor_idx] += bump
        return CapVolSurface(self.trade_date, self.strikes, self.tenors, self.maturities, vols)
    
    def output(self, tenors=True):
        return pd.DataFrame(self.vols, index=self.tenors if tenors else self.maturities, columns=self.strikes)
    
    def __repr__(self):
        return f"CapVolSurface({self.trade_date})"

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
    
    def caplet_vol(self, fixing_date, strike):
        target = (clean_date(fixing_date) - self.trade_date).astype(float) / 365
        target = np.clip(target, self.tfix.min(), self.tfix.max())
        strike = np.clip(strike, self.strikes.min(), self.strikes.max())
        
        smile = [np.sqrt(f(target) / target) for f in self.__tenor_interp]
        return PchipInterpolator(self.strikes, smile)(strike)
    
    def bump(self, tenor, bump=1):
        tenor = clean_tenor(tenor)
        idx = np.where(self.tenors == tenor.tenor)[0][0]
        vols = np.array(self.vols)
        vols[idx] += bump
        return CapletVolSurface(self.trade_date, self.strikes, self.tenors, self.fixings, vols)
    
    def output(self):
        return pd.DataFrame(self.vols, columns=self.strikes, index=self.tenors)
    
    def __repr__(self):
        return f"VolSurface({self.trade_date})"

class Market:
    def __init__(self,
            trade_date: Date,
            estr_curve: ZeroCurve,
            euribor_curve: ZeroCurve,
            caplet_surface: CapletVolSurface=None,
            cap_surface: CapVolSurface=None,
            ):
        
        self.trade_date = clean_date(trade_date)
        self.estr_curve = estr_curve
        self.euribor_curve = euribor_curve
        self.caplet_surface = caplet_surface
        self.cap_surface = cap_surface

    @classmethod
    def from_date(cls, trade_date, generate_caplets=True, generate_caps=True):
        trade_date  = clean_date(trade_date)
        z_estr      = ZeroCurve.from_date(trade_date, "ESTR")
        z_eur       = ZeroCurve.from_date(trade_date, "EURIBOR6M")
        caplet_vols = CapletVolSurface.from_date(trade_date) if generate_caplets else None
        cap_vols    = CapVolSurface.from_date(trade_date) if generate_caps else None
        return cls(trade_date, z_estr, z_eur, caplet_vols, cap_vols)
    
    def day_shift(self, other):
        return (self.trade_date - other.trade_date).astype("int")
    
    def __repr__(self):
        return f"Market({str(self.trade_date)})"


