import streamlit as st
import pandas as pd
from helpers import load_actual_pnl_data, load_sens_data, load_cap_stripping_data, load_vols_data, load_cap_validation_data, load_rates_data, load_zero_rates_data, load_curve_validation_data, load_hull_white_data
from risk import risk_tab
from vol import vol_tab
from curves import curve_tab
from conventions import conventions_tab
from pnl import pnl_tab
from hw import hull_white_tab

st.markdown("""
<style>
.block-container {
    padding: 2.5rem 2rem 2.5rem; max-width: 1600px;
}
</style>
""", unsafe_allow_html=True)

st.set_page_config(
    page_title            = "PnL Explain: FMM vs HW",
    layout                = "wide",
    initial_sidebar_state = "expanded",
    page_icon             = "👷🏼‍♀️"
)

# --------------------- #
# TITLE
# --------------------- #

st.markdown("""<h1 style='font-size: 1.5em;'>Explaining PnL in Interest Rate Options</h1>""", unsafe_allow_html=True)
st.caption("Greeks, Hedging and a Comparison of Hull–White and the Forward Market Model")
st.divider()

# --------------------- #
# DATA LOAD
# --------------------- #

actual_pnl_df = load_actual_pnl_data()
sens_df = load_sens_data()
vols_df = load_vols_data()
caplet_vol_df = load_cap_stripping_data()
cap_validation_df = load_cap_validation_data()
rates_df = load_rates_data()
zero_rates_df = load_zero_rates_data()
curve_validation_df = load_curve_validation_data()
hw_df = load_hull_white_data()

# --------------------- #
# TABS:
# 1. PnL Attribution
# 2. Greeks
# 3. Bumps
# --------------------- #

tab_1, tab_2, tab_3, tab_4, tab_5, tab_6 = st.tabs(["📈 &nbsp; PnL Attribution", "📊 &nbsp; Risk Panel", "📍 &nbsp; Vol Surface", "🔖 &nbsp; Hull White", "📉 &nbsp; Curves", "ℹ️ &nbsp; Market Conventions"])

@st.cache_data
def load_pnl():
    df = pd.read_parquet("../pricing_data/actual_pnl.parquet")
    return df

pnl = load_pnl()

# --------------------- #
# TAB 1 - PnL Attribution
# --------------------- #

with tab_1:
    pnl_tab(actual_pnl_df, sens_df)

# --------------------- #
# TAB 2 - GREEKS
# --------------------- #

with tab_2:
    risk_tab(sens_df)

# --------------------- #
# TAB 3 - Vol Surface
# --------------------- #

with tab_3:
    vol_tab(vols_df, caplet_vol_df, rates_df, cap_validation_df)

# --------------------- #
# TAB 4 - Hull-White
# --------------------- #

with tab_4:
    hull_white_tab(hw_df)

# --------------------- #
# TAB 5 – Curves
# --------------------- #

with tab_5:
    curve_tab(rates_df, zero_rates_df, curve_validation_df)

# --------------------- #
# TAB 6 – Conventions
# --------------------- #

with tab_6:
    conventions_tab()

# --------------------- #
# FOOTER
# --------------------- #
st.divider()
st.markdown(
"""<div style='padding: 0, margin: 0; font-size: 0.75em; text-align: center'><b>MSc Candidate:</b> Daniela Valentina Hidalgo Soto |
<b>Supervisor:</b> Prof. Francesco Rotondi<br>
Bocconi University · MAFINRISK · Final Work</div>""", unsafe_allow_html=True)