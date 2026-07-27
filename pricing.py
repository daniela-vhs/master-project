from instruments import Cap, IRS

import pandas as pd
import numpy as np

class Trade:
    def __init__(self, instrument, notional=1, position=1):
        self.instrument = instrument
        self.notional = notional
        self.position = position

    def value(self, market):
        return self.instrument.value(market) * self.notional * self.position

    def __repr__(self):
        position = "Pay" if self.position == 1 else "Rcv"
        return str(self.instrument) + f". Notional: {self.notional:0,.0f}. Position: {position}."

    def sens_boundaries(self, market):
        trade_maturity = market.trade_date + np.timedelta64(self.instrument.days_to_maturity(market))
        bounds = dict(Rate = dict(), Vol = dict())

        # ESTR boundaries
        estr_idx = np.where(pd.to_datetime(trade_maturity) <= pd.to_datetime(market.estr_curve.maturities))[0].min()
        estr_tenors = market.estr_curve.tenors[1 : estr_idx + 2]
        bounds["Rate"]["ESTR"] = estr_tenors

        # EURIBOR6M boundaries
        euribor_idx = np.where(pd.to_datetime(trade_maturity) <= pd.to_datetime(market.euribor_curve.maturities))[0].min()
        euribor_tenors = market.euribor_curve.tenors[1 : euribor_idx + 2]
        bounds["Rate"]["EURIBOR6M"] = euribor_tenors

        if isinstance(self.instrument, IRS):
            return bounds

        # Cap vol surface boundaries
        trade_strike = self.instrument.strike

        ## Rate
        eur_vol_rate_tenors = market.euribor_curve.tenors[1 : euribor_idx + 2]
        estr_vol_rate_tenors = market.estr_curve.tenors[1 : estr_idx + 2]
        bounds["Vol"]["RateTenors"] = dict()
        bounds["Vol"]["RateTenors"]["ESTR"] = estr_vol_rate_tenors
        bounds["Vol"]["RateTenors"]["EURIBOR6M"] = eur_vol_rate_tenors

        ## Tenors
        vol_tenor_idx = np.where(pd.to_datetime(trade_maturity) <= np.array(market.cap_surface.maturities))[0].min()
        vol_tenors = market.cap_surface.tenors[np.maximum(0, vol_tenor_idx - 4) : np.minimum(len(market.cap_surface.tenors), vol_tenor_idx + 2)]
        bounds["Vol"]["VolTenors"] = vol_tenors

        ## Strikes
        vol_strike_idx = (np.where(market.cap_surface.strikes < trade_strike)[0].max(), np.where(market.cap_surface.strikes > trade_strike)[0].min())
        vol_strikes = market.cap_surface.strikes[vol_strike_idx[0] - 1 : vol_strike_idx[1] + 2]
        bounds["Vol"]["VolStrikes"] = vol_strikes
        return bounds

    def delta(self, bumps, boundaries, curve, model="fmm", shifts=None):
        delta = []

        for tenor in boundaries["Rate"][curve]:
            bump    = bumps[curve][tenor]
            bump_bp = bumps[curve][tenor]["bp"]

            if isinstance(self.instrument, Cap) and model == "hw":
                p_up = self.instrument.value(bump["up"], "jamshidian_price", bump["up"].hull_white.a, bump["up"].hull_white.sigma)
                p_dn = self.instrument.value(bump["down"], "jamshidian_price", bump["down"].hull_white.a, bump["down"].hull_white.sigma)

            elif isinstance(self.instrument, IRS) or model == "fmm":
                p_up = self.instrument.value(bump["up"])
                p_dn = self.instrument.value(bump["down"])

            value = (p_up - p_dn) / (2 * bump_bp / 10_000)

            if abs(value) > 1e-10:
                delta_value = dict(
                    TradeDate = bump["mid"].trade_date,
                    Source = curve,
                    RateTenor = tenor,
                    Measure = "Delta",
                    Order = 1,
                    Value = value * self.notional * self.position
                )

                if shifts is not None:
                    delta_value["RateShift"] = shifts[curve][tenor]
                    delta_value["PnL"] = delta_value["RateShift"] * delta_value["Value"]

                delta.append(delta_value)

        return delta

    def gamma(self, bumps, boundaries, curve, model="fmm", shifts=None):
        gamma = []

        for tenor in boundaries["Rate"][curve]:
            bump    = bumps[curve][tenor]
            bump_bp = bumps[curve][tenor]["bp"]

            if isinstance(self.instrument, Cap) and model == "hw":
                p_up = self.instrument.value(bump["up"], "jamshidian_price", bump["up"].hull_white.a, bump["up"].hull_white.sigma)
                p_md = self.instrument.value(bump["mid"], "jamshidian_price", bump["mid"].hull_white.a, bump["mid"].hull_white.sigma)
                p_dn = self.instrument.value(bump["down"], "jamshidian_price", bump["down"].hull_white.a, bump["down"].hull_white.sigma)

            elif isinstance(self.instrument, IRS) or model == "fmm":
                p_up = self.instrument.value(bump["up"])
                p_md = self.instrument.value(bump["mid"])
                p_dn = self.instrument.value(bump["down"])

            value = (p_up - 2 * p_md + p_dn) / (bump_bp / 10_000) ** 2

            if abs(value) > 1e-10:
                gamma_value = dict(
                    TradeDate = bump["mid"].trade_date,
                    Source = curve,
                    RateTenor = tenor,
                    Measure = "Gamma",
                    Order = 2,
                    Value = value * self.notional * self.position
                )

                if shifts is not None:
                    gamma_value["RateShift"] = shifts[curve][tenor]
                    gamma_value["PnL"] = 0.5 * gamma_value["RateShift"] ** 2 * gamma_value["Value"]

                gamma.append(gamma_value)

        return gamma

    def vega(self, bumps, boundaries, model="fmm", shifts=None):
        vega = []

        if isinstance(self.instrument, IRS):
            return vega

        for strike in boundaries["Vol"]["VolStrikes"]:
            for tenor in boundaries["Vol"]["VolTenors"]:
                bump = bumps["CapVolSurface"][strike][tenor]

                if model == "hw":
                    p_up = self.instrument.value(bump["up"], "jamshidian_price", bump["up"].hull_white.a, bump["up"].hull_white.sigma)
                    p_dn = self.instrument.value(bump["down"], "jamshidian_price", bump["down"].hull_white.a, bump["down"].hull_white.sigma)
    
                elif model == "fmm":
                    p_up = self.instrument.value(bump["up"])
                    p_dn = self.instrument.value(bump["down"])
                
                value = (p_up - p_dn) / (2 * bump["bp"])
        
                if abs(value) > 1e-10:
                    vega_value = dict(
                        TradeDate = bump["mid"].trade_date,
                        Source = "Vol",
                        VolTenor = tenor,
                        Strike = strike,
                        Measure = "Vega",
                        Order = 1,
                        Value = value * self.notional * self.position
                    )

                    if shifts is not None:
                        vega_value["VolShift"] = shifts["CapVolSurface"][strike][tenor]
                        vega_value["PnL"] = vega_value["VolShift"] * vega_value["Value"]

                    vega.append(vega_value)

        return vega

    def volga(self, bumps, boundaries, model="fmm", shifts=None):
        volga = []

        if isinstance(self.instrument, IRS):
            return volga

        for strike in boundaries["Vol"]["VolStrikes"]:
            for tenor in boundaries["Vol"]["VolTenors"]:
                bump = bumps["CapVolSurface"][strike][tenor]

                if model == "hw":
                    p_up = self.instrument.value(bump["up"], "jamshidian_price", bump["up"].hull_white.a, bump["up"].hull_white.sigma)
                    p_md = self.instrument.value(bump["mid"], "jamshidian_price", bump["mid"].hull_white.a, bump["mid"].hull_white.sigma)
                    p_dn = self.instrument.value(bump["down"], "jamshidian_price", bump["down"].hull_white.a, bump["down"].hull_white.sigma)
    
                elif model == "fmm":
                    p_up = self.instrument.value(bump["up"])
                    p_md = self.instrument.value(bump["mid"])
                    p_dn = self.instrument.value(bump["down"])
                
                value = (p_up + p_dn - 2 * p_md) / (bump["bp"]) ** 2
        
                if abs(value) > 1e-10:
                    volga_value = dict(
                        TradeDate = bump["mid"].trade_date,
                        Source = "Vol",
                        VolTenor = tenor,
                        Strike = strike,
                        Measure = "Volga",
                        Order = 2,
                        Value = value * self.notional * self.position
                    )

                    if shifts is not None:
                        volga_value["VolShift"] = shifts["CapVolSurface"][strike][tenor]
                        volga_value["PnL"] = 0.5 * volga_value["VolShift"] ** 2 * volga_value["Value"]
                    
                    volga.append(volga_value)

        return volga

    def vanna(self, bumps, boundaries, curve, model="fmm", shifts=None):
        vanna = []

        if isinstance(self.instrument, IRS):
            return vanna

        for strike in boundaries["Vol"]["VolStrikes"][:2]:
            for vol_tenor in boundaries["Vol"]["VolTenors"][:2]:
                vbump = bumps["CapVolSurface"][strike][vol_tenor]
                vbp   = vbump["bp"]

                for rate_tenor in boundaries["Vol"]["RateTenors"][curve][:2]:
                    rbump = bumps[curve][rate_tenor]
                    rbp   = rbump["bp"]

                    rup_vup = self.instrument.rebuild_rates(rbump["up"]).rebuild_vol(vbump["up"])
                    rup_vdn = self.instrument.rebuild_rates(rbump["up"]).rebuild_vol(vbump["down"])
                    rdn_vup = self.instrument.rebuild_rates(rbump["down"]).rebuild_vol(vbump["up"])
                    rdn_vdn = self.instrument.rebuild_rates(rbump["down"]).rebuild_vol(vbump["down"])

                    if model == "fmm":
                        rup_vup = rup_vup.bachelier_price()
                        rup_vdn = rup_vdn.bachelier_price()
                        rdn_vup = rdn_vup.bachelier_price()
                        rdn_vdn = rdn_vdn.bachelier_price()

                    elif model == "hw":
                        rup_vup = rup_vup.jamshidian_price(vbump["up"].hull_white.a, vbump["up"].hull_white.sigma)
                        rup_vdn = rup_vdn.jamshidian_price(vbump["down"].hull_white.a, vbump["down"].hull_white.sigma)
                        rdn_vup = rdn_vup.jamshidian_price(vbump["up"].hull_white.a, vbump["up"].hull_white.sigma)
                        rdn_vdn = rdn_vdn.jamshidian_price(vbump["down"].hull_white.a, vbump["down"].hull_white.sigma)

                    value = (rup_vup - rup_vdn - rdn_vup + rdn_vdn) / (2 * rbp / 10_000) / (2 * vbp)

                    if abs(value) > 1e-10:
                        vanna_value = dict(
                            TradeDate = self.instrument.trade_date,
                            Source = curve,
                            RateTenor = rate_tenor,
                            VolTenor = vol_tenor,
                            Strike = strike,
                            Measure = "Vanna",
                            Order = 2,
                            Value = value * self.notional * self.position
                        )

                        if shifts is not None:
                            vanna_value["RateShift"] = shifts[curve][rate_tenor]
                            vanna_value["VolShift"] = shifts["CapVolSurface"][strike][vol_tenor]
                            vanna_value["PnL"] = vanna_value["Value"] * vanna_value["RateShift"] * vanna_value["VolShift"]

                        vanna.append(vanna_value)

        return vanna

    def theta(self, bumps, model="fmm", shifts=None):
        bump  = bumps["Time"]
        days  = (bump["up"].trade_date - self.instrument.trade_date).astype(int)

        if isinstance(self.instrument, IRS) or model == "fmm":
            value = (self.instrument.value(bump["up"]) - self.instrument.value(bump["mid"])) / days
        elif model == "hw":
            value = (self.instrument.value(bump["up"], "jamshidian_price", bump["up"].hull_white.a, bump["up"].hull_white.sigma) - self.instrument.value(bump["mid"], "jamshidian_price", bump["mid"].hull_white.a, bump["mid"].hull_white.sigma)) / days

        theta_value = dict(
            TradeDate = self.instrument.trade_date,
            Source = "Time",
            Measure = "Theta",
            Order = 1,
            Value = value * self.notional * self.position,
        )

        if shifts is not None:
            theta_value["DayShift"] = shifts["Time"]
            theta_value["PnL"] = theta_value["DayShift"] * theta_value["Value"]

        return [theta_value]

    def sens(self, bumps, boundaries, model="fmm", shifts=None):
        sens = []
        sens.extend(self.delta(bumps, boundaries, "EURIBOR6M", model, shifts))
        sens.extend(self.delta(bumps, boundaries, "ESTR", model, shifts))
        sens.extend(self.gamma(bumps, boundaries, "EURIBOR6M", model, shifts))
        sens.extend(self.gamma(bumps, boundaries, "ESTR", model, shifts))
        sens.extend(self.theta(bumps, model, shifts))

        if isinstance(self.instrument, IRS):
            return sens

        sens.extend(self.vega(bumps, boundaries, model, shifts))
        sens.extend(self.volga(bumps, boundaries, model, shifts))
        sens.extend(self.vanna(bumps, boundaries, "EURIBOR6M", model, shifts))
        sens.extend(self.vanna(bumps, boundaries, "ESTR", model, shifts))
        return sens

    def actual_pnl(self, mkt_start, mkt_end):
        return self.value(mkt_end) - self.value(mkt_start)

class Portfolio:
    def __init__(self, trades: list[Trade] = []):
        self.trades = trades

    def append(self, trade):
        self.trades.append(trade)

    def value(self, market):
        mkt_value = 0
        for trade in self.trades:
            mkt_value += trade.value(market)
        return mkt_value

    def __repr__(self):
        return "Portfolio:\n" + "\n".join([f"- {str(i)}" for i in self.trades])


