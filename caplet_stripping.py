from market import Market
from instruments import Cap, Caplet, IRS
from curves import schedule_generation
from dates import day_count_fraction, Tenor, clean_date, clean_tenor

from scipy.optimize import brentq

import pandas as pd
import numpy as np

from typing import TypeAlias

Date: TypeAlias = np.datetime64

vols = pd.read_parquet("clean_data/vols.parquet").set_index(["Date", "Tenor", "IsATM", "Strike"]).sort_index()

# Cap Generator
def generate_cap(trade_date: Date, tenor: Tenor, market: Market, strike=None):
    trade_date = clean_date(trade_date)
    tenor      = clean_tenor(tenor)
    anchors    = [Tenor("3Y") + Tenor(f"{i * 6}M") for i in range(15) if Tenor("3Y") + Tenor(f"{i * 6}M") <= tenor]

    # Curves
    z_euribor = market.euribor_curve
    z_estr    = market.estr_curve

    # Cap Properties
    atm = True

    if strike is not None:
        atm = False
    else:
        irs = IRS.from_benchmark("EURIBOR6M", trade_date, tenor)
        strike = irs.par_rate(market)

    # Caplets
    schedule = schedule_generation(trade_date, tenor, "SemiAnnual", pay_delay="0 Business Days", include_tenor=True).set_index("Tenor")
    schedule = schedule[schedule.FixingDate > trade_date]
    cap_vol  = market.cap_surface.cap_vol(schedule.iloc[-1].AccrualEnd, strike)

    schedule["TradeDate"]  = trade_date
    schedule["Tau"]        = day_count_fraction(schedule.AccrualStart, schedule.AccrualEnd, "ACT/360")
    schedule["TFix"]       = day_count_fraction(schedule.TradeDate, schedule.FixingDate, "ACT/365")
    schedule["EurStartDF"] = z_euribor[schedule.AccrualStart.to_numpy()]
    schedule["EurEndDF"]   = z_euribor[schedule.AccrualEnd.to_numpy()]
    schedule["EstrPayDF"]  = z_estr[schedule.PaymentDate.to_numpy()]
    schedule["FwdRate"]    = (schedule.EurStartDF / schedule.EurEndDF - 1) / schedule.Tau
    
    caplet_dict = dict()

    for t, row in schedule.iterrows():
        bucket = [i.tenor for i in anchors if Tenor(t) <= i][0]
        caplet = Caplet(trade_date, t, bucket, row.FixingDate, row.AccrualStart, row.AccrualEnd, row.PaymentDate, row.Tau, row.TFix, row.EurStartDF, row.EurEndDF, row.EstrPayDF, row.FwdRate, cap_vol)
        caplet_dict[t] = caplet

    return Cap(trade_date, tenor, cap_vol, strike, caplet_dict)

# Objective Function
def objective(guess, cap):
    for caplet in cap.caplets.values():
        if caplet.cap_tenor_bucket == cap.tenor:
            caplet.caplet_vol = guess

    return cap.bachelier_price() - cap.bachelier_price(flat_vol=True)

# Stripping Algorithm
def caplet_stripping(trade_date, market, strike=None):
    anchors       = [(Tenor("3Y") + Tenor(f"{i * 6}M")).tenor for i in range(15)]
    stripped_vols = {}

    for anchor in anchors[:]:
        cap = generate_cap(trade_date, anchor, market, strike=strike)
        
        for vol_tenor, vol in stripped_vols.items():
            for caplet in cap.caplets.values():
                if vol_tenor == caplet.cap_tenor_bucket.tenor:
                    caplet.caplet_vol = vol[0]

        # print(cap.caplets)
        result                = brentq(objective, 10, 500, args=(cap), full_output=True)
        stripped_vols[anchor] = (result[0], cap)

    return stripped_vols

# Output Stripping in table
def stripping_output(trade_date, market, strike=None):
    data = caplet_stripping(clean_date(trade_date), market, strike=strike)
    df   = pd.DataFrame(data.keys(), columns=["Tenor"])

    df["TradeDate"]   = trade_date
    df["StrippedVol"] = [i[0] for i in data.values()]
    df["Strike"]      = [i[1].strike for i in data.values()]
    df["IsATM"]       = True if strike is None else False
    df.Tenor          = df.Tenor.astype("category")
    df.TradeDate      = pd.to_datetime(df.TradeDate)

    return df[["TradeDate", "Tenor", "StrippedVol", "Strike", "IsATM"]]

# Stripping for all Strikes
def stripping_output_full(trade_date):
    trade_date = clean_date(trade_date)
    market = Market.from_date(trade_date, generate_caplets=False)
    strikes   = [-0.005, -0.0025, -0.00125, 0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, None]
    return pd.concat([stripping_output(trade_date, market, strike) for strike in strikes])

# Output Caplets in table
def caplet_output(trade_date, market):
    data = generate_cap(trade_date, "10Y", market).caplets
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
    if rebuild:
        stripping_output_full("2021-07-12").to_parquet("clean_data/cap_stripping.parquet", index=False)

    base = pd.read_parquet("clean_data/cap_stripping.parquet")
    dates = pd.date_range(base.TradeDate.max(), vols.index.get_level_values(0).max())[1:]

    new = []

    for date in dates[:]:
        try:
            new.append(stripping_output_full(date))
            print(f"{date.date()}: OK.")
        except:
            continue

    if len(new) > 0:
        new = pd.concat([base] + new)
        new.to_parquet("clean_data/cap_stripping.parquet", index=False)
        print("Data updated.")

    else:
        new = base.copy()
        print("No new data.")

def caplet_master_loop(rebuild=False):
    if rebuild:
        caplet_output("2021-07-12", Market.from_date("2021-07-12")).to_parquet("clean_data/caplets.parquet", index=False)

    base = pd.read_parquet("clean_data/caplets.parquet")
    dates = pd.date_range(base.TradeDate.max(), vols.index.get_level_values(0).max())[1:]

    new = []

    for date in dates[:]:
        try:
            new.append(caplet_output(date, Market.from_date(date)))
            print(f"{date.date()}: OK.")
        except:
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


