from instruments import Cap, Caplet
from dates import Tenor

from scipy.optimize import brentq

import pandas as pd
import numpy as np

from typing import TypeAlias

Date: TypeAlias = np.datetime64

vols = pd.read_parquet("clean_data/vols.parquet").set_index(["Date", "Tenor", "IsATM", "Strike"]).sort_index()

# Objective Function
def objective(guess, cap, target):
    for caplet in cap.caplets.values():
        if caplet.cap_tenor_bucket == cap.tenor:
            caplet.caplet_vol = guess

    return cap.bachelier_price() - target

# Stripping Algorithm
def caplet_stripping(market, strike=None):
    anchors       = [(Tenor("3Y") + Tenor(f"{i * 6}M")).tenor for i in range(15)]
    stripped_vols = {}
    all_caplets   = Caplet.generate(market)

    for anchor in anchors[:]:
        cap = Cap.generate(market, anchor, strike, True, all_caplets)
        
        for vol_tenor, vol in stripped_vols.items():
            for caplet in cap.caplets.values():
                if vol_tenor == caplet.cap_tenor_bucket.tenor:
                    caplet.caplet_vol = vol[0]

        target                = cap.bachelier_price(flat_vol=True)
        result                = brentq(objective, 0.1, 500, args=(cap, target), full_output=True)
        stripped_vols[anchor] = (result[0], cap)

    return stripped_vols

# Output Stripping in table
def stripping_output(market, strike=None):
    data = caplet_stripping(market, strike=strike)
    df   = pd.DataFrame(data.keys(), columns=["Tenor"])

    df["TradeDate"]   = market.trade_date
    df["StrippedVol"] = [i[0] for i in data.values()]
    df["Strike"]      = [i[1].strike for i in data.values()]
    df["IsATM"]       = True if strike is None else False
    df.Tenor          = df.Tenor.astype("category")
    df.TradeDate      = pd.to_datetime(df.TradeDate)

    return df[["TradeDate", "Tenor", "StrippedVol", "Strike", "IsATM"]]

# Stripping for all Strikes
def stripping_output_full(market):
    strikes   = [-0.005, -0.0025, -0.00125, 0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, None]
    return pd.concat([stripping_output(market, strike) for strike in strikes])

# Output Caplets in table
def caplet_output(market):
    data = Caplet.generate(market)
    caplets = list(data.values())
    df = pd.DataFrame([i.trade_date for i in caplets], columns=["TradeDate"])
    df["FixingDate"] = [i.fixing_date for i in caplets]
    df["AccrualStart"] = [i.accrual_start for i in caplets]
    df["AccrualEnd"] = [i.accrual_end for i in caplets]
    df["PaymentDate"] = [i.payment_date for i in caplets]
    df["Tenor"] = [i.tenor.tenor for i in caplets]
    df["CapTenorBucket"] = [i.cap_tenor_bucket.tenor for i in caplets]
    df["Tau"] = [i.tau for i in caplets]
    df["TFix"] = [i.tfix for i in caplets]
    df["EurStartDF"] = [i.eur_start_df for i in caplets]
    df["EurEndDF"] = [i.eur_end_df for i in caplets]
    df["EstrPayDF"] = [i.estr_pay_df for i in caplets]
    df["FwdRate"] = [i.fwd_rate for i in caplets]
    df.Tenor = df.Tenor.astype("category")
    df.CapTenorBucket = df.CapTenorBucket.astype("category")
    return df

def cap_stripping_loop(rebuild=False):
    from market import Market
    base = pd.read_parquet("clean_data/cap_stripping.parquet")
    
    if rebuild:
        base = stripping_output_full(Market.from_date("2021-07-12", generate_caplets=False, generate_hw=False))
        print(f"2021-07-12: OK.")

    dates = vols.index.get_level_values("Date")
    dates = dates[dates > base.TradeDate.max()].unique()

    new = []

    for date in dates[:]:
        try:
            mkt = Market.from_date(date, generate_caplets=False, generate_hw=False)
            new.append(stripping_output_full(mkt))
            print(f"{date.date()}: OK.")
        except Exception as e:
            print(f"{date.date()}: Error - {e}")

    if len(new) > 0:
        new = pd.concat([base] + new)
        new.to_parquet("clean_data/cap_stripping.parquet", index=False)
        print("Data updated.")

    else:
        new = base.copy()
        print("No new data.")

def caplet_master_loop(rebuild=False):
    from market import Market
    base = pd.read_parquet("clean_data/caplets.parquet")
    
    if rebuild:
        base = caplet_output(Market.from_date("2021-07-12", generate_caplets=False, generate_caps=False, generate_hw=False))

    dates = vols.index.get_level_values("Date")
    dates = dates[dates > base.TradeDate.max()].unique()
    
    new = []

    for date in dates[:]:
        try:
            new.append(caplet_output(Market.from_date(date, generate_caplets=False, generate_caps=False, generate_hw=False)))
            print(f"{date.date()}: OK.")
        except Exception as e:
            print(f"{date.date()}: Error - {e}.")
            continue

    if len(new) > 0:
        new = pd.concat([base] + new)
        new.to_parquet("clean_data/caplets.parquet", index=False)
        print("Data updated.")

    else:
        new = base.copy()
        print("No new data.")

if __name__ == "__main__":
    cap_stripping_loop()
    caplet_master_loop()


