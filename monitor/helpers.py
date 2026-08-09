import streamlit as st
import pandas as pd
import warnings
from quant.dates import Tenor

warnings.filterwarnings("ignore")

@st.cache_data
def load_vols_data():
    return pd.read_parquet("../clean_data/vols.parquet")

@st.cache_data
def load_cap_stripping_data():
    return pd.read_parquet("../clean_data/cap_stripping.parquet")

@st.cache_data
def load_rates_data():
    return pd.read_parquet("../clean_data/rates.parquet")

@st.cache_data
def load_zero_rates_data():
    return pd.read_parquet("../clean_data/zero_rates.parquet")

@st.cache_data
def load_hull_white_data():
    return pd.read_parquet("../clean_data/hw_calibration.parquet")

@st.cache_data
def load_actual_pnl_data():
    return pd.read_parquet("../pricing_data/actual_pnl.parquet")

@st.cache_data
def load_sens_data():
    return pd.read_parquet("../pricing_data/sens.parquet")

@st.cache_data
def load_cap_validation_data():
    return pd.read_parquet("../validation/cap_repricing_validation.parquet")

@st.cache_data
def load_curve_validation_data():
    return pd.read_parquet("../validation/irs_repricing.parquet")

def date_slider(dates, key, label="Valuation date"):
    dates = sorted(dates)
    default_idx = 0
    selected = st.select_slider(
        label,
        options=dates,
        value=dates[default_idx],
        format_func=lambda d: pd.Timestamp(d).strftime("%d %b %Y"),
        label_visibility="visible",
        key=key,
    )
    return pd.Timestamp(selected)

def get_trades(sens_data):
    df = sens_data.drop_duplicates(subset=["TradeDate", "Instrument", "TradeTenor", "TradeMaturity", "IsATM", "TradeStrike", "Notional", "Position"])
    return df[["TradeDate", "Instrument", "TradeTenor", "TradeMaturity", "IsATM", "TradeStrike", "Notional", "Position"]].to_dict("records")

def print_trade(trade):
    return f'{trade["Instrument"]}({trade["TradeDate"].date()}, {trade["TradeTenor"]})'

def get_risk_table(trade, sens_data):
    return sens_data[
        (sens_data.Instrument == trade["Instrument"]) &
        (sens_data.TradeTenor == trade["TradeTenor"]) &
        (sens_data.TradeDate == trade["TradeDate"]) &
        (sens_data.TradeStrike == trade["TradeStrike"])
        ]

def get_trade_dates(trade, sens_data):
    return get_risk_table(trade, sens_data).ValueDate.unique()

def sort_tenor(df, name):
    return df.reindex(sorted([i for i in df.index.get_level_values(level=name).unique()], key=lambda x: Tenor(x)), level=name)

def sort_index_tenor(df):
    return df.reindex(sorted(df.index, key=lambda x: Tenor(x)))

def apply_heatmap(df, cmap, num_format="{:0,.2f}", axis=0, low=0.2, high=0.2):
    df = df.style\
           .background_gradient(cmap=cmap, low=low, high=high, axis=axis)\
           .format(num_format)
    return df

def write_subtitle(content):
    st.markdown(
        f"""<h4 style='font-size: 1.3em;'>
            {content}
        </h4>""",
        unsafe_allow_html=True
    )

def write_subsubtitle(content):
    st.markdown(
        f"""<h4 style='font-size: 1.1em;'>
            {content}
        </h4>""",
        unsafe_allow_html=True
    )

def chart_layout():
    return dict(
        margin = dict(
            t = 0,
            b = 0,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            xanchor="left",
            x=0,
            y=1,
        )
    )