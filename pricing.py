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

    def _price(self, market, model="fmm", cache=None): # AI Optimization
        """Price the instrument on a bump market, memoized per Market object within one sens pass."""
        key = id(market)
        if cache is not None and key in cache:
            return cache[key]
        if isinstance(self.instrument, Cap) and model == "hw":
            p = self.instrument.value(market, "jamshidian_price", market.hull_white.a, market.hull_white.sigma)
        else:
            p = self.instrument.value(market)
        if cache is not None:
            cache[key] = p
        return p

    def sens_boundaries(self, market):
        trade_maturity = self.instrument.maturity
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
        vol_strikes = market.cap_surface.strikes[np.maximum(0, vol_strike_idx[0] - 1) : vol_strike_idx[1] + 2]
        bounds["Vol"]["VolStrikes"] = vol_strikes
        return bounds

    def delta(self, bumps, boundaries, curve, model="fmm", shifts=None, cache=None):
        delta = []

        for tenor in boundaries["Rate"][curve]:
            bump    = bumps[curve][tenor]
            bump_bp = bumps[curve][tenor]["bp"]

            p_up = self._price(bump["up"],   model, cache)
            p_dn = self._price(bump["down"], model, cache)

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

    def gamma(self, bumps, boundaries, curve, model="fmm", shifts=None, cache=None):
        gamma = []

        for tenor in boundaries["Rate"][curve]:
            bump    = bumps[curve][tenor]
            bump_bp = bumps[curve][tenor]["bp"]

            p_up = self._price(bump["up"],   model, cache)
            p_dn = self._price(bump["down"], model, cache)
            p_md = self._price(bump["mid"],  model, cache)

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

    def vega(self, bumps, boundaries, model="fmm", shifts=None, cache=None):
        vega = []

        if isinstance(self.instrument, IRS):
            return vega

        for strike in boundaries["Vol"]["VolStrikes"]:
            for tenor in boundaries["Vol"]["VolTenors"]:
                bump = bumps["CapVolSurface"][strike][tenor]

                p_up = self._price(bump["up"],   model, cache)
                p_dn = self._price(bump["down"], model, cache)
                
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

    def volga(self, bumps, boundaries, model="fmm", shifts=None, cache=None):
        volga = []

        if isinstance(self.instrument, IRS):
            return volga

        for strike in boundaries["Vol"]["VolStrikes"]:
            for tenor in boundaries["Vol"]["VolTenors"]:
                bump = bumps["CapVolSurface"][strike][tenor]

                p_up = self._price(bump["up"],   model, cache)
                p_dn = self._price(bump["down"], model, cache)
                p_md = self._price(bump["mid"],  model, cache)
                
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

        for strike in boundaries["Vol"]["VolStrikes"][:]:
            for vol_tenor in boundaries["Vol"]["VolTenors"][:]:
                vbump = bumps["CapVolSurface"][strike][vol_tenor]
                vbp   = vbump["bp"]

                for rate_tenor in boundaries["Vol"]["RateTenors"][curve][:]:
                    rbump = bumps[curve][rate_tenor]
                    rbp   = rbump["bp"]

                    rup_vup = self.instrument.rebuild_rates(rbump["up"]).rebuild_vol(vbump["up"])
                    rup_vdn = self.instrument.rebuild_rates(rbump["up"]).rebuild_vol(vbump["down"])
                    rdn_vup = self.instrument.rebuild_rates(rbump["down"]).rebuild_vol(vbump["up"])
                    rdn_vdn = self.instrument.rebuild_rates(rbump["down"]).rebuild_vol(vbump["down"])

                    if model == "fmm":
                        # p_up = self.instrument.value(bump["up"])
                        rup_vup = rup_vup.bachelier_price(vbump["up"])
                        rup_vdn = rup_vdn.bachelier_price(vbump["down"])
                        rdn_vup = rdn_vup.bachelier_price(vbump["up"])
                        rdn_vdn = rdn_vdn.bachelier_price(vbump["down"])

                    elif model == "hw":
                        # p_up = self.instrument.value(bump["up"], "jamshidian_price", bump["up"].hull_white.a, bump["up"].hull_white.sigma)
                        rup_vup = rup_vup.jamshidian_price(vbump["up"], vbump["up"].hull_white.a, vbump["up"].hull_white.sigma)
                        rup_vdn = rup_vdn.jamshidian_price(vbump["down"], vbump["down"].hull_white.a, vbump["down"].hull_white.sigma)
                        rdn_vup = rdn_vup.jamshidian_price(vbump["up"], vbump["up"].hull_white.a, vbump["up"].hull_white.sigma)
                        rdn_vdn = rdn_vdn.jamshidian_price(vbump["down"], vbump["down"].hull_white.a, vbump["down"].hull_white.sigma)

                    value = (rup_vup - rup_vdn - rdn_vup + rdn_vdn) / (2 * rbp / 10_000) / (2 * vbp)

                    if abs(value) > 1e-10:
                        vanna_value = dict(
                            TradeDate = rbump["mid"].trade_date,
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

    def vanna(self, bumps, boundaries, curve, model="fmm", shifts=None): # AI Optimization
        vanna = []

        if isinstance(self.instrument, IRS):
            return vanna

        # Precompute caplet-vol vectors once per (strike, vol_tenor, direction):
        # they depend only on the bumped caplet surface, not on the rate bump.
        base    = bumps["Time"]["mid"]
        ref     = self.instrument.rebuild_market(base)
        live_t  = [t for t, c in ref.caplets.items() if c.status(base) == "Live"]
        fixings = [ref.caplets[t].fixing_date for t in live_t]

        vol_cache = dict()
        for strike in boundaries["Vol"]["VolStrikes"]:
            for vol_tenor in boundaries["Vol"]["VolTenors"]:
                vbump = bumps["CapVolSurface"][strike][vol_tenor]
                vol_cache[(strike, vol_tenor, "up")]   = np.atleast_1d(vbump["up"].caplet_surface.caplet_vol(fixings, self.instrument.strike))
                vol_cache[(strike, vol_tenor, "down")] = np.atleast_1d(vbump["down"].caplet_surface.caplet_vol(fixings, self.instrument.strike))

        def price(cap, vols, vmarket, a=None, sigma=None):
            for n, t in enumerate(live_t):
                cap.caplets[t].caplet_vol = vols[n]
            if model == "hw":
                return cap.jamshidian_price(vmarket, a, sigma)
            return cap.bachelier_price(vmarket)

        for rate_tenor in boundaries["Vol"]["RateTenors"][curve]:
            rbump = bumps[curve][rate_tenor]
            rbp   = rbump["bp"]

            # Rebuild the rate legs ONCE per rate tenor. rebuild_market (not rebuild_rates)
            # so tfix is measured from the evaluation date, not from inception.
            r_up = self.instrument.rebuild_market(rbump["up"])
            r_dn = self.instrument.rebuild_market(rbump["down"])

            for strike in boundaries["Vol"]["VolStrikes"]:
                for vol_tenor in boundaries["Vol"]["VolTenors"]:
                    vbump = bumps["CapVolSurface"][strike][vol_tenor]
                    vbp   = vbump["bp"]
                    v_up  = vol_cache[(strike, vol_tenor, "up")]
                    v_dn  = vol_cache[(strike, vol_tenor, "down")]

                    if model == "hw":
                        au, su = vbump["up"].hull_white.a,   vbump["up"].hull_white.sigma
                        ad, sd = vbump["down"].hull_white.a, vbump["down"].hull_white.sigma
                        rup_vup = price(r_up, v_up, vbump["up"],   au, su)
                        rup_vdn = price(r_up, v_dn, vbump["down"], ad, sd)
                        rdn_vup = price(r_dn, v_up, vbump["up"],   au, su)
                        rdn_vdn = price(r_dn, v_dn, vbump["down"], ad, sd)
                    else:
                        rup_vup = price(r_up, v_up, vbump["up"])
                        rup_vdn = price(r_up, v_dn, vbump["down"])
                        rdn_vup = price(r_dn, v_up, vbump["up"])
                        rdn_vdn = price(r_dn, v_dn, vbump["down"])

                    value = (rup_vup - rup_vdn - rdn_vup + rdn_vdn) / (2 * rbp / 10_000) / (2 * vbp)

                    if abs(value) > 1e-10:
                        vanna_value = dict(
                            TradeDate = rbump["mid"].trade_date,
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
        days  = (bump["up"].trade_date - bump["mid"].trade_date).astype(int)

        if isinstance(self.instrument, IRS) or model == "fmm":
            value = (self.instrument.value(bump["up"]) - self.instrument.value(bump["mid"])) / days
        elif model == "hw":
            value = (self.instrument.value(bump["up"], "jamshidian_price", bump["up"].hull_white.a, bump["up"].hull_white.sigma) - self.instrument.value(bump["mid"], "jamshidian_price", bump["mid"].hull_white.a, bump["mid"].hull_white.sigma)) / days

        theta_value = dict(
            TradeDate = bump["mid"].trade_date,
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
        cache = dict()
        sens.extend(self.delta(bumps, boundaries, "EURIBOR6M", model, shifts, cache))
        sens.extend(self.delta(bumps, boundaries, "ESTR", model, shifts, cache))
        sens.extend(self.gamma(bumps, boundaries, "EURIBOR6M", model, shifts, cache))
        sens.extend(self.gamma(bumps, boundaries, "ESTR", model, shifts, cache))
        sens.extend(self.theta(bumps, model, shifts))

        if isinstance(self.instrument, IRS):
            return sens

        sens.extend(self.vega(bumps, boundaries, model, shifts, cache))
        sens.extend(self.volga(bumps, boundaries, model, shifts, cache))
        sens.extend(self.vanna(bumps, boundaries, "EURIBOR6M", model, shifts))
        sens.extend(self.vanna(bumps, boundaries, "ESTR", model, shifts))
        return sens

    def actual_pnl(self, market_start, market_end):
        return self.value(market_end) - self.value(market_start)

    def split_pnl(self, current_market, previous_market):
        cap  = self.instrument.rebuild_market(previous_market)
        base = cap.bachelier_price(previous_market) * self.notional * self.position

        rate_only = cap.rebuild_rates(current_market, status_market=previous_market).bachelier_price(current_market, status_market=previous_market) * self.notional * self.position - base
        vol_only  = cap.rebuild_vol(current_market, status_market=previous_market).bachelier_price(current_market, status_market=previous_market)   * self.notional * self.position - base
        time      = cap.rebuild_time(current_market).bachelier_price(current_market) * self.notional * self.position - base

        total_cap = cap.rebuild_market(current_market)
        total     = total_cap.bachelier_price(current_market) * self.notional * self.position - base
        realized  = total_cap.realized(current_market) * self.notional * self.position
        cross     = total - rate_only - vol_only - time

        return [dict(
            ValueDate = current_market.trade_date, PrevDate = previous_market.trade_date,
            RatePnL = rate_only, VolPnL = vol_only, TimePnL = time, CrossPnL = cross,
            TotalPnL = total, RealizedPnL = realized,
        )]

    def pnl_attribution(self, sens):
        pnl_explain = {
            "Total" : 0
        }

        for row in sens:
            measure = row["Measure"]
            source = row["Source"]
            value = row["PnL"]
            pnl_explain["Total"] += value

            if measure not in pnl_explain:
                pnl_explain[measure] = dict()

            if source not in pnl_explain[measure]:
                pnl_explain[measure][source] = value

            else:
                pnl_explain[measure][source] += value

        return pnl_explain

class Portfolio:
    def __init__(self, trades: list[Trade] = None):
        self.trades = trades if trades is not None else []

    def append(self, trade):
        self.trades.append(trade)

    def value(self, market):
        mkt_value = 0
        for trade in self.trades:
            mkt_value += trade.value(market)
        return mkt_value

    def __repr__(self):
        return "Portfolio:\n" + "\n".join([f"- {str(i)}" for i in self.trades])

    def generate_bumps(self, market):
        max_maturity = np.array([i.instrument.maturity for i in self.trades]).max()
        strikes      = np.array([i.instrument.strike for i in self.trades])
        min_strike   = strikes.min()
        max_strike   = strikes.max()
        return market.generate_bumps(max_maturity, min_strike, max_strike)

    def sens(self, market, bumps, model="fmm", shifts=None):
        sens = []

        for trade in self.trades:
            bounds = trade.sens_boundaries(market)
            sens.extend(trade.sens(bumps, bounds, model, shifts))

        return sens

    def actual_pnl(self, market_start, market_end):
        return self.value(market_end) - self.value(market_start)

    def delta(self, bumps, boundaries, curve, model="fmm", shifts=None):
        delta = []

        for trade in self.trades:
            delta.extend(trade.delta(bumps, boundaries, curve, model, shifts))

        return delta

    def gamma(self, bumps, boundaries, curve, model="fmm", shifts=None):
        gamma = []

        for trade in self.trades:
            gamma.extend(trade.gamma(bumps, boundaries, curve, model, shifts))

        return gamma

    def vega(self, bumps, boundaries, model="fmm", shifts=None):
        vega = []
        
        for trade in self.trades:
            vega.extend(trade.vega(bumps, boundaries, model, shifts))

        return vega

    def volga(self, bumps, boundaries, model="fmm", shifts=None):
        volga = []
                
        for trade in self.trades:
            volga.extend(trade.volga(bumps, boundaries, model, shifts))

        return volga

    def vanna(self, bumps, boundaries, curve, model="fmm", shifts=None):
        vanna = []
                        
        for trade in self.trades:
            vanna.extend(trade.vanna(bumps, boundaries, curve, model, shifts))

        return vanna

    def theta(self, bumps, model="fmm", shifts=None):
        theta = []
                                
        for trade in self.trades:
            theta.extend(trade.theta(bumps, model, shifts))

        return theta


