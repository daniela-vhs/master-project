from instruments import Cap
from scipy.optimize import minimize
import numpy as np
import pandas as pd
from dates import clean_date
from curves import buscal

hw_data = pd.read_parquet("clean_data/hw_calibration.parquet").set_index("TradeDate")

class HullWhite:
    def __init__(self, trade_date: np.datetime64, a: float, sigma: float, error: float = 0, at_bound: bool = False):
        self.trade_date = clean_date(trade_date)
        self.a = a
        self.sigma = sigma
        self.error = error
        self.at_bound = at_bound

    @classmethod
    def build(cls, date):
        return cls(date, 0.03, 0.01)

    @classmethod
    def from_date(cls, date):
        data = hw_data.loc[date].to_dict()
        return cls(date, data["a"], data["sigma"], data["ResidualError"], data["AtBound"])

    def __target_price(self, cap_list):
        return {tenor: cap.bachelier_price() for tenor, cap in cap_list.items()}

    def __hw_error(self, params, cap_list, targets):
        sse = 0
        for cap in cap_list.values():
            sse += (targets[cap.tenor.tenor] - cap.jamshidian_price(params[0], params[1])) ** 2
        return sse

    def __minimizer(self, cap_list, targets, random_start = False):
        # Bounds
        a_bound     = (0.001, 1)
        sigma_bound = (0.0001, None)

        # Starting point
        if random_start:
            guess = (10 ** np.random.uniform(np.log10(0.005), np.log10(0.5)),
                     np.random.uniform(0.003, 0.03))
        else:
            guess = (self.a, self.sigma)

        # Optimizer
        result = minimize(self.__hw_error, guess, args=(cap_list, targets), bounds=(a_bound, sigma_bound), method="L-BFGS-B")
        params = result.x
        error  = result.fun

        # Check if result matches the bounds
        tol = 1e-8
        at_bound = (abs(params[0] - a_bound[0]) < tol) or (abs(params[0] - a_bound[1]) < tol) or (abs(params[1] - sigma_bound[0]) < tol)

        return dict(
            a = params[0],
            sigma = params[1],
            error = error,
            at_bound = at_bound
        )

    def calibrate(self, market, n_tests = 0):
        caps    = Cap.from_market(market)["ATM"]
        targets = self.__target_price(caps)
        best    = self.__minimizer(caps, targets, False)

        for _ in range(n_tests):
            if "error" not in best:
                best = self.__minimizer(caps, targets, True)

            else:
                new = self.__minimizer(caps, targets, True)
                if new["error"] < best["error"]:
                    best = new

        return HullWhite(
            trade_date = market.trade_date,
            a = best["a"],
            sigma = best["sigma"],
            error = best["error"],
            at_bound = best["at_bound"]
        )

    def initial_calibrate(self, market, rel_tol_a=0.02, rel_tol_sigma=0.01, patience=3, max_tests=50):
        caps    = Cap.from_market(market)["ATM"]
        targets = self.__target_price(caps)

        best         = self.__minimizer(caps, targets, random_start=True)
        stable_count = 0
        n_tests      = 1

        while stable_count < patience and n_tests < max_tests:
            new = self.__minimizer(caps, targets, random_start=True)
            n_tests += 1

            if new["error"] < best["error"]:
                best         = new
                stable_count = 0
            else:
                close = (abs(new["a"] - best["a"]) / max(best["a"], 1e-6) < rel_tol_a) and \
                        (abs(new["sigma"] - best["sigma"]) / max(best["sigma"], 1e-6) < rel_tol_sigma)
                stable_count = stable_count + 1 if close else 0

        return HullWhite(
            trade_date = market.trade_date,
            a          = best["a"],
            sigma      = best["sigma"],
            error      = best["error"],
            at_bound   = best["at_bound"]
        )

    def output(self):
        return pd.DataFrame([dict(
            TradeDate = self.trade_date,
            a = self.a,
            sigma = self.sigma,
            ResidualError = self.error,
            AtBound = self.at_bound
        )])

    def __repr__(self):
        return f"HullWhite(a: {self.a:.6f}, sigma: {(self.sigma * 10_000):.4f} bp)"

def recalibration_loop(start_date = None, end_date = None, n_tests = 3):
    from market import Market
    dates = hw_data.index.unique()

    if start_date is not None:
        dates = dates[dates >= start_date]

    if end_date is not None:
        dates = dates[dates <= end_date]

    new_base = []

    for date in dates[:]:
        try:
            original = HullWhite.from_date(date)
            market   = Market.from_date(date)
            new = original.calibrate(market, n_tests)

            new_base.append(new.output())
            print(f"{date.date()}: OK.")

        except Exception as e:
            print(f"(ERROR) {date.date()}: {e}.")

    if len(new_base) > 0:
        output = pd.concat([hw_data.reset_index()] + new_base).drop_duplicates(subset=["TradeDate"], keep="last").sort_values(by=["TradeDate"])
        output.to_parquet("clean_data/hw_calibration.parquet", index=False)
        print("Data updated.")

    else:
        print("No new data.")

def hw_loop(rebuild=False):
    from market import Market
    base = pd.read_parquet("clean_data/hw_calibration.parquet")
    vols = pd.read_parquet("clean_data/vols.parquet")

    if rebuild:
        mkt  = Market.from_date("2021-07-12")
        base = HullWhite.build("2021-07-12").calibrate(mkt, 10).output()
        print(f"2021-07-12: OK.")

    dates = pd.bdate_range(base.TradeDate.max(), vols.Date.max(), holidays=buscal.holidays, freq="C")[1:]

    new = []

    for date in dates[:]:
        try:
            mkt = Market.from_date(date)
            new.append(HullWhite.build(date).calibrate(mkt, 10).output())
            print(f"{date.date()}: OK.")
        except Exception as e:
            print(f"{date.date()}: Exception – {e}.")
            continue

    if len(new) > 0:
        new = pd.concat([base] + new)
        new.to_parquet("clean_data/hw_calibration.parquet", index=False)
        print("Data updated.")
    else:
        new = base.copy()
        print("No new data.")

if __name__ == "__main__":
    hw_loop()


