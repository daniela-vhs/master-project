import pandas as pd

# Clean rates.xlsx
# Input: Excel file
# Output: Parquet long file
with pd.ExcelFile("raw_data/rates.xlsx") as f:
    estr = pd.read_excel(f, sheet_name="ESTR", index_col=0, header=[0, 1])\
        .rename_axis("Date").rename_axis(["Tenor", "Ticker"], axis=1)
    euribor = pd.read_excel(f, sheet_name="EURIBOR 6M", index_col=0, header=[0, 1])\
        .rename_axis("Date").rename_axis(["Tenor", "Ticker"], axis=1)
    index = pd.read_excel(f, sheet_name="INDEX", index_col=0)\
        .rename_axis("Date").rename_axis("Ticker", axis=1)

# Clean ESTR Curve
estr_clean = estr.dropna().unstack().to_frame("Rate")
estr_clean["Curve"] = "ESTR"
estr_clean.reset_index(inplace=True)
min_date = estr_clean.Date.min()
max_date = estr_clean.Date.max()

# Clean EURIBOR 6M Curve
euribor_clean = euribor.dropna().unstack().to_frame("Rate")
euribor_clean["Curve"] = "EURIBOR6M"
euribor_clean.reset_index(inplace=True)
min_date = max(min_date, euribor_clean.Date.min())
max_date = min(max_date, euribor_clean.Date.max())

# Clean Index
index_clean = index.dropna().unstack().to_frame("Rate")
index_clean["Curve"] = index_clean.apply(lambda x: "EURIBOR6M" if "EUR006" in x.name[0] else "ESTR", axis=1)
index_clean["Tenor"] = "1D"
index_clean.reset_index(inplace=True)
min_date = max(min_date, index_clean.Date.min())
max_date = min(max_date, index_clean.Date.max())

# Rates output
rates_output = pd.concat([estr_clean, euribor_clean, index_clean]).drop("Ticker", axis=1)\
    .set_index(["Date", "Curve", "Tenor"]).sort_index().reset_index()
rates_output["Curve"] = rates_output["Curve"].astype("category")
rates_output["Tenor"] = rates_output["Tenor"].astype("category")
rates_output["Rate"] = pd.to_numeric(rates_output["Rate"], errors="coerce")
rates_output = rates_output[(rates_output.Date >= min_date) & (rates_output.Date <= max_date)].reset_index(drop=True)
rates_output.to_parquet("clean_data/rates.parquet")

# Clean vol.xlsx
# Input: Excel file
# Output: Parquet long file
with pd.ExcelFile("raw_data/vols.xlsx") as f:
    vol_data = [pd.DataFrame()]
    
    for strike in f.sheet_names:
        df = pd.read_excel(f, sheet_name=strike, index_col=[0, 1], header=[0, 1]).dropna()\
            .droplevel(0).droplevel(0, axis=1).rename_axis("Date").rename_axis("Tenor", axis=1)\
            .stack().to_frame("Vol").reset_index()
        df["Strike"] = float(strike.replace("%", "")) if strike != "ATM" else float("nan")
        df["IsATM"] = True if strike == "ATM" else False # ATM is date-dependent, not fixed
        df["Tenor"] = df.Tenor.apply(lambda x: f"{x}Y").astype("category")
        vol_data.append(df)

    vol_data = pd.concat(vol_data).set_index(["Date", "Tenor", "Strike"]).sort_index().reset_index()

vol_data.to_parquet("clean_data/vols.parquet")


