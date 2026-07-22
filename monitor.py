import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os

st.set_page_config(
    page_title="Thesis Monitor — Greeks & PnL: HW vs FMM",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
[data-testid="stSidebar"],
.main, .stApp {
    background-color: #ffffff !important;
    color: #111827 !important;
    color-scheme: light !important;
}
.block-container { padding: 2.5rem 2rem 1.5rem; max-width: 1600px; }
header[data-testid="stHeader"] { background: transparent; }
div[data-testid="stDecoration"] { display: none; }
.thesis-header { border-bottom: 1.5px solid #1a1a2e; padding-bottom: 0.75rem; margin-bottom: 1.25rem; }
.thesis-title  { font-size: 1.05rem; font-weight: 600; color: #1a1a2e; letter-spacing: -0.01em; margin: 0; }
.thesis-sub    { font-size: 0.72rem; color: #6b7280; font-family: 'JetBrains Mono', monospace; margin: 0.15rem 0 0; }
.section-label { font-size: 0.62rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #9ca3af; margin-bottom: 0.4rem; display: block; }
.pill { display: inline-block; background: #f0f4ff; color: #1e3a8a; border-radius: 4px; padding: 2px 8px; font-size: 0.68rem; font-family: 'JetBrains Mono', monospace; font-weight: 500; margin-right: 6px; }
.pill-warn { background: #fef2f2; color: #991b1b; }
.vol-table { border-collapse: collapse; width: 100%; white-space: nowrap; font-family: 'JetBrains Mono', monospace; font-size: 0.67rem; }
.vol-table th { color: #6b7280; text-align: right; padding: 4px 8px; border-bottom: 1px solid #e5e7eb; font-weight: 500; }
.vol-table td { text-align: right; padding: 3px 8px; color: #374151; }
.vol-table tr:hover td { background: #f9fafb; }
.vol-table .tc { font-weight: 600; color: #1a1a2e; text-align: left !important; }
.vol-table .atm-col { color: #1e3a8a; font-weight: 600; }
.vol-table .atm-rate { color: #6b7280; font-size: 0.6rem; }
.conv-card { background: #f9fafb; border: 0.5px solid #e5e7eb; border-radius: 8px; padding: 1rem 1.25rem; margin-bottom: 0.75rem; }
.conv-title { font-size: 0.8rem; font-weight: 600; color: #1a1a2e; margin-bottom: 0.5rem; }
.conv-row { display: flex; justify-content: space-between; font-size: 0.72rem; padding: 3px 0; border-bottom: 0.5px solid #f3f4f6; }
.conv-key { color: #6b7280; }
.conv-val { color: #111827; font-family: 'JetBrains Mono', monospace; font-weight: 500; }
hr.thin { border: none; border-top: 0.5px solid #e5e7eb; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_zero_rates():
    df = pd.read_parquet("clean_data/zero_rates.parquet")
    df["TradeDate"] = pd.to_datetime(df["TradeDate"])
    df["Maturity"]  = pd.to_datetime(df["Maturity"])
    return df

@st.cache_data
def load_vols():
    df = pd.read_parquet("clean_data/vols.parquet")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

@st.cache_data
def load_stripped_vols():
    df = pd.read_parquet("clean_data/cap_stripping.parquet")
    df["TradeDate"] = pd.to_datetime(df["TradeDate"])
    df["Tenor"] = df["Tenor"].astype(str)
    return df

@st.cache_data
def load_rates():
    df = pd.read_parquet("clean_data/rates.parquet")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

@st.cache_data
def load_irs_repricing():
    df = pd.read_parquet("validation/irs_repricing.parquet")
    df["TradeDate"] = pd.to_datetime(df["TradeDate"])
    df["Tenor"] = df["Tenor"].astype(str)
    return df

@st.cache_data
def load_cap_repricing_validation():
    df = pd.read_parquet("validation/cap_repricing_validation.parquet")
    df["Date"] = pd.to_datetime(df["Date"])
    df["Tenor"] = df["Tenor"].astype(str)
    return df

@st.cache_data
def load_hw_calibration():
    df = pd.read_parquet("clean_data/hw_calibration.parquet")
    df["TradeDate"] = pd.to_datetime(df["TradeDate"])
    return df

@st.cache_data
def load_conventions():
    convs = {}
    for name, fname in [("ESTR", "estr.json"), ("EURIBOR6M", "euribor6m.json"), ("VOL_SURFACE", "vol_surface.json")]:
        path = os.path.join("market_conventions", fname)
        if os.path.exists(path):
            with open(path) as f:
                convs[name] = json.load(f)
    return convs

zero_rates     = load_zero_rates()
vols           = load_vols()
stripped_vols  = load_stripped_vols()
rates_df       = load_rates()
irs_repricing  = load_irs_repricing()
cap_repricing  = load_cap_repricing_validation()
hw_calib       = load_hw_calibration()
conventions    = load_conventions()

all_dates   = sorted(zero_rates["TradeDate"].unique())

# Tenors common to the flat surface, the stripped surface and the cap-repricing
# validation set — this project's locked scope (3Y-10Y annual cap tenors).
COMMON_TENORS = ["3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y"]

# Strikes common to both surfaces. The stripped (caplet) surface lacks +0.125%
# and +0.25%, since those columns are only quoted against Euribor 3M.
_flat_strikes  = set(vols.loc[~vols["IsATM"], "Strike"].dropna().unique())
_strip_strikes = set((stripped_vols.loc[~stripped_vols["IsATM"], "Strike"].dropna().unique() * 100).round(3))
COMMON_STRIKES = sorted(_flat_strikes & _strip_strikes)
COMMON_STRIKE_LABELS = [f"{s:+.3f}%" if s != 0.0 else "0.000%" for s in COMMON_STRIKES]

# ── Helpers ───────────────────────────────────────────────────────────────────
def date_select_slider(dates, key, default_frac=0.6, label="Trade date"):
    """A slider that steps discretely through actual dates (not a bare index)."""
    dates = sorted(dates)
    default_idx = min(int(len(dates) * default_frac), len(dates) - 1)
    selected = st.select_slider(
        label,
        options=dates,
        value=dates[default_idx],
        format_func=lambda d: pd.Timestamp(d).strftime("%d %b %Y"),
        label_visibility="collapsed",
        key=key,
    )
    return pd.Timestamp(selected)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="thesis-header">
  <p class="thesis-title">Greeks & PnL Explained: Hull-White vs FMM</p>
  <p class="thesis-sub">EUR cap/caplet pricing · Euribor 6M · ESTR OIS discounting · Normal (Bachelier) vols</p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_curves, tab_vols, tab_hw, tab_conv = st.tabs(
    ["Zero curves", "Vol surface", "Hull-White", "Market conventions"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ZERO CURVES
# ══════════════════════════════════════════════════════════════════════════════
with tab_curves:
    selected_date = date_select_slider(all_dates, key="date_slider_curves")

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        st.markdown(f'<span class="pill">📅 {selected_date.strftime("%d %b %Y")}</span>',
                    unsafe_allow_html=True)

    st.markdown("<div style='margin:0.75rem 0;'></div>", unsafe_allow_html=True)

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<span class="section-label">Curve selector</span>', unsafe_allow_html=True)
        curve_choice = st.radio("Curve", ["ESTR", "EURIBOR6M", "Both"],
                                index=2, horizontal=True, label_visibility="collapsed")

        day_data = zero_rates[zero_rates["TradeDate"] == selected_date].copy()
        day_data["DaysOffset"] = (day_data["Maturity"] - selected_date).dt.days
        day_data["ZeroRate"]   = np.where(
            day_data["DaysOffset"] > 0,
            -np.log(day_data["DiscountFactor"]) / (day_data["DaysOffset"] / 365),
            np.nan
        )

        fig = go.Figure()
        colors = {"ESTR": "#1E90FF", "EURIBOR6M": "#FF6347"}
        labels = {"ESTR": "ESTR OIS", "EURIBOR6M": "Euribor 6M IRS"}
        curves_to_plot = ["ESTR", "EURIBOR6M"] if curve_choice == "Both" else [curve_choice]

        for curve in curves_to_plot:
            cd = day_data[day_data["Curve"] == curve].dropna(subset=["ZeroRate"]).sort_values("Maturity")
            if cd.empty:
                continue
            fig.add_trace(go.Scatter(
                x=cd["Maturity"], y=cd["ZeroRate"] * 100,
                mode="lines+markers", name=labels[curve],
                line=dict(color=colors[curve], width=2),
                marker=dict(size=5, color=colors[curve]),
                hovertemplate="<b>%{x|%d %b %Y}</b><br>%{y:.4f}%<extra></extra>"
            ))

        fig.update_layout(
            height=320, margin=dict(l=0, r=0, t=8, b=0),
            paper_bgcolor="white", plot_bgcolor="white",
            font=dict(family="Inter", size=11),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=10)),
            xaxis=dict(showgrid=True, gridcolor="#f3f4f6", gridwidth=0.5,
                       tickformat="%Y", title=dict(text="Maturity", font=dict(size=10, color="#9ca3af")),
                       showline=True, linecolor="#e5e7eb"),
            yaxis=dict(showgrid=True, gridcolor="#f3f4f6", gridwidth=0.5,
                       title=dict(text="Zero rate (%)", font=dict(size=10, color="#9ca3af")),
                       ticksuffix="%", tickformat=".2f", showline=True, linecolor="#e5e7eb"),
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with right:
        st.markdown('<span class="section-label">Discount factors & zero rates</span>', unsafe_allow_html=True)
        show = day_data[day_data["Curve"].isin(curves_to_plot)].copy()
        show = show[show["Tenor"] != "2D"].copy()
        show["Zero Rate (%)"] = (show["ZeroRate"] * 100).round(4)
        show["DiscountFactor"] = show["DiscountFactor"].round(6)
        show["Maturity"] = show["Maturity"].dt.strftime("%Y-%m-%d")
        show = show[["Curve", "Tenor", "Maturity", "DiscountFactor", "Zero Rate (%)"]].reset_index(drop=True)
        st.dataframe(show, use_container_width=True, height=340)

        # Historical zero rate for one tenor
        st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
        st.markdown('<span class="section-label">Historical zero rate — single tenor</span>', unsafe_allow_html=True)

        available_tenors_estr = zero_rates[zero_rates["Curve"] == "ESTR"]["Tenor"].unique().tolist()
        hist_tenor = st.selectbox("Tenor", sorted(available_tenors_estr, key=lambda x: float(x[:-1]) * (12 if x[-1]=="Y" else 1)),
                                  index=4, label_visibility="collapsed")
        hist_curve = st.radio("", ["ESTR", "EURIBOR6M"], horizontal=True, label_visibility="collapsed", key="hist_curve")

        hist = zero_rates[(zero_rates["Curve"] == hist_curve) & (zero_rates["Tenor"] == hist_tenor)].copy()
        hist["DaysOffset"] = (hist["Maturity"] - hist["TradeDate"]).dt.days
        hist["ZeroRate"]   = -np.log(hist["DiscountFactor"]) / (hist["DaysOffset"] / 365) * 100

        fig3 = go.Figure(go.Scatter(
            x=hist["TradeDate"], y=hist["ZeroRate"],
            mode="lines", line=dict(color=colors.get(hist_curve, "#1E90FF"), width=1.5),
            hovertemplate="%{x|%d %b %Y}: %{y:.4f}%<extra></extra>"
        ))
        fig3.update_layout(
            height=200, margin=dict(l=0, r=0, t=8, b=0),
            paper_bgcolor="white", plot_bgcolor="white",
            font=dict(family="Inter", size=10),
            xaxis=dict(showgrid=True, gridcolor="#f3f4f6", gridwidth=0.5,
                       showline=True, linecolor="#e5e7eb"),
            yaxis=dict(showgrid=True, gridcolor="#f3f4f6", gridwidth=0.5,
                       ticksuffix="%", tickformat=".2f", showline=True, linecolor="#e5e7eb"),
        )
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    # ── Curve bootstrap validation (IRS repricing) ──────────────────────────
    st.markdown("<hr class='thin'>", unsafe_allow_html=True)
    st.markdown('<span class="section-label">Bootstrap validation — IRS repricing error</span>', unsafe_allow_html=True)
    st.caption("Feeding each curve's own bootstrapped instruments back through the pricer. Error = recovered par rate − quoted rate, in bp.")

    val_left, val_right = st.columns([3, 2], gap="large")

    with val_left:
        agg = (irs_repricing
               .groupby("TradeDate")[["EstrError", "EurError"]]
               .apply(lambda x: x.abs().max())
               .reset_index())

        figv = make_subplots(specs=[[{"secondary_y": True}]])
        figv.add_trace(go.Scatter(
            x=agg["TradeDate"], y=agg["EstrError"], name="ESTR (left)",
            mode="lines", line=dict(color="#1E90FF", width=1.2),
            hovertemplate="%{x|%d %b %Y}: %{y:.5f} bp<extra>ESTR</extra>"
        ), secondary_y=False)
        figv.add_trace(go.Scatter(
            x=agg["TradeDate"], y=agg["EurError"], name="Euribor 6M (right)",
            mode="lines", line=dict(color="#FF6347", width=1.2),
            hovertemplate="%{x|%d %b %Y}: %{y:.5f} bp<extra>Euribor 6M</extra>"
        ), secondary_y=True)

        figv.update_layout(
            height=280, margin=dict(l=0, r=0, t=8, b=0),
            paper_bgcolor="white", plot_bgcolor="white",
            font=dict(family="Inter", size=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=10)),
            xaxis=dict(showgrid=True, gridcolor="#f3f4f6", showline=True, linecolor="#e5e7eb"),
            hovermode="x unified"
        )
        figv.update_yaxes(title_text="ESTR error (bp)", ticksuffix=" bp", secondary_y=False,
                           showgrid=True, gridcolor="#f3f4f6", showline=True, linecolor="#e5e7eb")
        figv.update_yaxes(title_text="Euribor 6M error (bp)", ticksuffix=" bp", secondary_y=True,
                           showgrid=False, showline=True, linecolor="#e5e7eb")
        st.plotly_chart(figv, use_container_width=True, config={"displayModeBar": False})
        st.caption("Max absolute repricing error across all tenors, per trade date.")

    with val_right:
        day_val = irs_repricing[irs_repricing["TradeDate"] == selected_date].copy()
        if day_val.empty:
            st.info("No IRS repricing data for this date.")
        else:
            day_val = day_val.sort_values("Tenor", key=lambda s: s.map(lambda t: float(t[:-1])))
            day_val["EstrError (bp)"] = day_val["EstrError"].round(6)
            day_val["EurError (bp)"]  = day_val["EurError"].round(6)
            st.dataframe(day_val[["Tenor", "EstrError (bp)", "EurError (bp)"]].reset_index(drop=True),
                         use_container_width=True, height=280)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — VOL SURFACE
# ══════════════════════════════════════════════════════════════════════════════
with tab_vols:
    # Only dates present in BOTH the flat and the stripped surface can be compared
    vol_dates = sorted(set(vols["Date"].unique()) & set(stripped_vols["TradeDate"].unique()))

    selected_date_v = date_select_slider(vol_dates, key="date_slider_vols")
    st.markdown(f'<span class="pill">📅 {selected_date_v.strftime("%d %b %Y")}</span>',
                unsafe_allow_html=True)
    st.markdown("<div style='margin:0.75rem 0;'></div>", unsafe_allow_html=True)

    day_vols   = vols[vols["Date"] == selected_date_v].copy()
    day_strip  = stripped_vols[stripped_vols["TradeDate"] == selected_date_v].copy()

    otm_vols   = day_vols[~day_vols["IsATM"]].copy()
    otm_strip  = day_strip[~day_strip["IsATM"]].copy()

    strikes_sorted = COMMON_STRIKES
    strike_labels  = COMMON_STRIKE_LABELS

    def strip_lookup(tenor, strike_pct):
        """Stripped vol for a cap tenor and a flat-surface strike quoted in %."""
        target = strike_pct / 100
        cell = otm_strip[(otm_strip["Tenor"] == tenor) & np.isclose(otm_strip["Strike"], target)]
        return float(cell["StrippedVol"].iloc[0]) if not cell.empty else np.nan

    def flat_lookup(tenor, strike_pct):
        cell = otm_vols[(otm_vols["Tenor"] == tenor) & (otm_vols["Strike"] == strike_pct)]
        return float(cell["Vol"].iloc[0]) if not cell.empty else np.nan

    # ── Build matrices (flat vs stripped) over the common tenor scope ───────
    z_flat, z_strip = [], []
    for tenor in COMMON_TENORS:
        z_flat.append([flat_lookup(tenor, s) for s in strikes_sorted])
        z_strip.append([strip_lookup(tenor, s) for s in strikes_sorted])

    z_flat_arr  = np.array(z_flat, dtype=float)
    z_strip_arr = np.array(z_strip, dtype=float)
    zmin = np.nanmin([np.nanmin(z_flat_arr), np.nanmin(z_strip_arr)])
    zmax = np.nanmax([np.nanmax(z_flat_arr), np.nanmax(z_strip_arr)])
    colorscale = "Viridis"

    st.markdown('<span class="section-label">Flat vs stripped vol surface — bp (Bachelier), 3Y-10Y scope</span>',
                unsafe_allow_html=True)

    hm_left, hm_right = st.columns(2, gap="large")

    for col, title, z in [(hm_left, "Flat (quoted)", z_flat_arr), (hm_right, "Stripped (caplet)", z_strip_arr)]:
        with col:
            st.markdown(f'<span class="section-label">{title}</span>', unsafe_allow_html=True)
            z_text = [[f"{v:.1f}" if not np.isnan(v) else "" for v in row] for row in z]
            figh = go.Figure(go.Heatmap(
                z=z, x=strike_labels, y=COMMON_TENORS,
                zmin=zmin, zmax=zmax, colorscale=colorscale,
                text=z_text, texttemplate="%{text}",
                textfont=dict(size=9, family="JetBrains Mono"),
                hovertemplate="Tenor: %{y}<br>Strike: %{x}<br>Vol: %{z:.1f} bp<extra></extra>",
                showscale=True, colorbar=dict(title=dict(text="bp", font=dict(size=10)), thickness=10, len=0.9),
                xgap=1, ygap=1
            ))
            figh.update_layout(
                height=320, margin=dict(l=0, r=30, t=8, b=0),
                paper_bgcolor="white", plot_bgcolor="white",
                font=dict(family="Inter", size=10),
                xaxis=dict(title=dict(text="Strike", font=dict(size=10, color="#9ca3af")),
                           tickangle=-45, type="category"),
                yaxis=dict(title=dict(text="Cap tenor", font=dict(size=10, color="#9ca3af")),
                           autorange="reversed", type="category"),
            )
            st.plotly_chart(figh, use_container_width=True, config={"displayModeBar": False})

    # ── 3D comparison ─────────────────────────────────────────────────────
    st.markdown("<div style='margin-top:0.5rem;'></div>", unsafe_allow_html=True)
    st.markdown('<span class="section-label">3D vol surface comparison</span>', unsafe_allow_html=True)

    tenor_idx = list(range(len(COMMON_TENORS)))
    fig3d = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "surface"}, {"type": "surface"}]],
        subplot_titles=("Flat", "Stripped"),
        horizontal_spacing=0.02
    )
    fig3d.add_trace(go.Surface(
        z=z_flat_arr, x=strikes_sorted, y=tenor_idx,
        cmin=zmin, cmax=zmax, colorscale=colorscale, showscale=False,
        hovertemplate="Strike: %{x:.3f}%<br>Tenor idx: %{y}<br>Vol: %{z:.1f} bp<extra>Flat</extra>"
    ), row=1, col=1)
    fig3d.add_trace(go.Surface(
        z=z_strip_arr, x=strikes_sorted, y=tenor_idx,
        cmin=zmin, cmax=zmax, colorscale=colorscale,
        colorbar=dict(title=dict(text="bp", font=dict(size=10)), thickness=10, x=1.02),
        hovertemplate="Strike: %{x:.3f}%<br>Tenor idx: %{y}<br>Vol: %{z:.1f} bp<extra>Stripped</extra>"
    ), row=1, col=2)

    scene_common = dict(
        xaxis=dict(title="Strike (%)", tickformat=".2f"),
        yaxis=dict(title="Tenor", tickvals=tenor_idx, ticktext=COMMON_TENORS),
        zaxis=dict(title="Vol (bp)", range=[zmin, zmax]),
        camera=dict(eye=dict(x=1.8, y=-1.8, z=0.8))
    )
    fig3d.update_layout(
        height=420, margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor="white", font=dict(family="Inter, sans-serif", size=10),
        scene=scene_common, scene2=scene_common
    )

    # Render as a raw HTML component (not st.plotly_chart) so we can attach a
    # JS listener that mirrors camera rotation/zoom/pan between the two scenes.
    div_id = "vol3d_sync"
    plot_html = fig3d.to_html(include_plotlyjs="cdn", full_html=False, div_id=div_id)
    font_import = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
    body { font-family: 'Inter', sans-serif; margin: 0; }
    </style>
    """
    sync_script = f"""
    <script>
    (function() {{
        var gd = document.getElementById("{div_id}");
        var syncing = false;
        gd.on("plotly_relayout", function(eventData) {{
            if (syncing) return;
            var update = {{}};
            var changed = false;
            if (eventData["scene.camera"]) {{
                update["scene2.camera"] = eventData["scene.camera"];
                changed = true;
            }} else if (eventData["scene2.camera"]) {{
                update["scene.camera"] = eventData["scene2.camera"];
                changed = true;
            }}
            if (changed) {{
                syncing = true;
                Plotly.relayout(gd, update).then(function() {{ syncing = false; }});
            }}
        }});
    }})();
    </script>
    """
    components.html(font_import + plot_html + sync_script, height=440, scrolling=False)

    # ── Historical evolution: flat vs stripped, per tenor & strike ─────────
    st.markdown("<div style='margin-top:0.5rem;'></div>", unsafe_allow_html=True)
    st.markdown('<span class="section-label">Historical evolution — flat vs stripped</span>', unsafe_allow_html=True)

    ev_c1, ev_c2 = st.columns(2)
    with ev_c1:
        ev_tenor = st.selectbox("Cap tenor", COMMON_TENORS, index=4, key="ev_tenor")
    with ev_c2:
        ev_strike_label = st.selectbox("Strike", ["ATM"] + strike_labels, index=0, key="ev_strike")

    if ev_strike_label == "ATM":
        flat_hist  = vols[(vols["IsATM"]) & (vols["Tenor"] == ev_tenor)].sort_values("Date")
        strip_hist = stripped_vols[(stripped_vols["IsATM"]) & (stripped_vols["Tenor"] == ev_tenor)].sort_values("TradeDate")
        strip_x, strip_y = strip_hist["TradeDate"], strip_hist["StrippedVol"]
    else:
        ev_strike_val = strikes_sorted[strike_labels.index(ev_strike_label)]
        flat_hist  = vols[(~vols["IsATM"]) & (vols["Tenor"] == ev_tenor) & (vols["Strike"] == ev_strike_val)].sort_values("Date")
        strip_hist = stripped_vols[(~stripped_vols["IsATM"]) & (stripped_vols["Tenor"] == ev_tenor)
                                    & np.isclose(stripped_vols["Strike"], ev_strike_val / 100)].sort_values("TradeDate")
        strip_x, strip_y = strip_hist["TradeDate"], strip_hist["StrippedVol"]

    fig_ev = go.Figure()
    fig_ev.add_trace(go.Scatter(
        x=flat_hist["Date"], y=flat_hist["Vol"], name="Flat",
        mode="lines", line=dict(color="#FF6347", width=1.3),
        hovertemplate="%{x|%d %b %Y}: %{y:.1f} bp<extra>Flat</extra>"
    ))
    fig_ev.add_trace(go.Scatter(
        x=strip_x, y=strip_y, name="Stripped",
        mode="lines", line=dict(color="#1E90FF", width=1.3),
        hovertemplate="%{x|%d %b %Y}: %{y:.1f} bp<extra>Stripped</extra>"
    ))
    fig_ev.update_layout(
        height=260, margin=dict(l=0, r=0, t=8, b=0),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Inter", size=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=10)),
        xaxis=dict(showgrid=True, gridcolor="#f3f4f6", showline=True, linecolor="#e5e7eb"),
        yaxis=dict(showgrid=True, gridcolor="#f3f4f6", ticksuffix=" bp",
                   title=dict(text="Normal vol", font=dict(size=10, color="#9ca3af")),
                   showline=True, linecolor="#e5e7eb"),
        hovermode="x unified"
    )
    st.plotly_chart(fig_ev, use_container_width=True, config={"displayModeBar": False})
    st.caption(f"{ev_tenor} cap, strike {ev_strike_label} — flat (quoted, cap-level) vs stripped (caplet-level) normal vol.")

    # ── Cap repricing validation ────────────────────────────────────────────
    st.markdown("<hr class='thin'>", unsafe_allow_html=True)
    st.markdown('<span class="section-label">Cap repricing validation — flat vol price vs stripped vol price</span>', unsafe_allow_html=True)
    st.caption("Repricing a cap with stripped caplet vols and comparing to the flat-vol price. Error in bp, split ATM/non-ATM (different scales).")

    cr_left, cr_right = st.columns(2, gap="large")

    agg_non_atm = (cap_repricing[~cap_repricing["IsATM"]]
                   .groupby("Date")["RepricingError"].apply(lambda x: x.abs().max()).reset_index())
    agg_atm = (cap_repricing[cap_repricing["IsATM"]]
               .groupby("Date")["RepricingError"].apply(lambda x: x.abs().max()).reset_index())

    with cr_left:
        st.markdown('<span class="section-label">Non-ATM — max error</span>', unsafe_allow_html=True)
        fign = go.Figure(go.Scatter(
            x=agg_non_atm["Date"], y=agg_non_atm["RepricingError"],
            mode="lines", line=dict(color="#1E90FF", width=1.2),
            hovertemplate="%{x|%d %b %Y}: %{y:.2e} bp<extra></extra>"
        ))
        fign.update_layout(
            height=230, margin=dict(l=0, r=0, t=8, b=0),
            paper_bgcolor="white", plot_bgcolor="white",
            font=dict(family="Inter", size=10),
            xaxis=dict(showgrid=True, gridcolor="#f3f4f6", showline=True, linecolor="#e5e7eb"),
            yaxis=dict(showgrid=True, gridcolor="#f3f4f6", ticksuffix=" bp", showline=True, linecolor="#e5e7eb"),
        )
        st.plotly_chart(fign, use_container_width=True, config={"displayModeBar": False})

    with cr_right:
        st.markdown('<span class="section-label">ATM — max error</span>', unsafe_allow_html=True)
        figa = go.Figure(go.Scatter(
            x=agg_atm["Date"], y=agg_atm["RepricingError"],
            mode="lines", line=dict(color="#FF6347", width=1.2),
            hovertemplate="%{x|%d %b %Y}: %{y:.3f} bp<extra></extra>"
        ))
        figa.update_layout(
            height=230, margin=dict(l=0, r=0, t=8, b=0),
            paper_bgcolor="white", plot_bgcolor="white",
            font=dict(family="Inter", size=10),
            xaxis=dict(showgrid=True, gridcolor="#f3f4f6", showline=True, linecolor="#e5e7eb"),
            yaxis=dict(showgrid=True, gridcolor="#f3f4f6", ticksuffix=" bp", showline=True, linecolor="#e5e7eb"),
        )
        st.plotly_chart(figa, use_container_width=True, config={"displayModeBar": False})

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — HULL-WHITE
# ══════════════════════════════════════════════════════════════════════════════
with tab_hw:
    hw_dates = sorted(hw_calib["TradeDate"].unique())
    selected_date_hw = date_select_slider(hw_dates, key="date_slider_hw")

    row_hw = hw_calib[hw_calib["TradeDate"] == selected_date_hw]
    if not row_hw.empty:
        r = row_hw.iloc[0]
        bound_pill = '<span class="pill pill-warn">⚠ at bound</span>' if r["AtBound"] else '<span class="pill">interior</span>'
        st.markdown(
            f'<span class="pill">📅 {selected_date_hw.strftime("%d %b %Y")}</span>'
            f'<span class="pill">a = {r["a"]:.4f}</span>'
            f'<span class="pill">σ_HW = {r["sigma"]*10000:.1f} bp</span>'
            f'<span class="pill">SSE = {r["ResidualError"]:.2e}</span>'
            f'{bound_pill}',
            unsafe_allow_html=True
        )
    st.markdown("<div style='margin:0.75rem 0;'></div>", unsafe_allow_html=True)

    hw_calib_sorted = hw_calib.sort_values("TradeDate").copy()
    hw_calib_sorted["sigma_bp"] = hw_calib_sorted["sigma"] * 10000
    bound_colors = np.where(hw_calib_sorted["AtBound"], "#FF6347", "#1E90FF")

    fig_a = go.Figure()
    fig_a.add_trace(go.Scatter(
        x=hw_calib_sorted["TradeDate"], y=hw_calib_sorted["a"],
        mode="markers", marker=dict(size=3, color=bound_colors),
        hovertemplate="%{x|%d %b %Y}: a=%{y:.4f}<extra></extra>"
    ))
    fig_a.add_vline(x=selected_date_hw, line=dict(color="#9ca3af", width=1, dash="dot"))
    fig_a.update_layout(
        height=250, margin=dict(l=0, r=0, t=8, b=0),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Inter", size=10),
        xaxis=dict(showgrid=True, gridcolor="#f3f4f6", showline=True, linecolor="#e5e7eb"),
        yaxis=dict(title=dict(text="a — mean reversion speed", font=dict(size=10, color="#9ca3af")),
                   showgrid=True, gridcolor="#f3f4f6", showline=True, linecolor="#e5e7eb"),
    )

    fig_s = go.Figure()
    fig_s.add_trace(go.Scatter(
        x=hw_calib_sorted["TradeDate"], y=hw_calib_sorted["sigma_bp"],
        mode="markers", marker=dict(size=3, color=bound_colors),
        hovertemplate="%{x|%d %b %Y}: σ=%{y:.1f} bp<extra></extra>"
    ))
    fig_s.add_vline(x=selected_date_hw, line=dict(color="#9ca3af", width=1, dash="dot"))
    fig_s.update_layout(
        height=250, margin=dict(l=0, r=0, t=8, b=0),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Inter", size=10),
        xaxis=dict(showgrid=True, gridcolor="#f3f4f6", showline=True, linecolor="#e5e7eb"),
        yaxis=dict(title=dict(text="σ_HW (bp)", font=dict(size=10, color="#9ca3af")), ticksuffix=" bp",
                   showgrid=True, gridcolor="#f3f4f6", showline=True, linecolor="#e5e7eb"),
    )

    hw_c1, hw_c2 = st.columns(2, gap="large")
    with hw_c1:
        st.markdown('<span class="section-label">a — mean-reversion speed</span>', unsafe_allow_html=True)
        st.plotly_chart(fig_a, use_container_width=True, config={"displayModeBar": False})
    with hw_c2:
        st.markdown('<span class="section-label">σ_HW — instantaneous volatility</span>', unsafe_allow_html=True)
        st.plotly_chart(fig_s, use_container_width=True, config={"displayModeBar": False})

    st.caption("Red points = calibration pinned at a bound (mostly the pre-2022 negative-rate window). Dotted line marks the selected date.")

    st.markdown("<div style='margin-top:0.5rem;'></div>", unsafe_allow_html=True)
    st.markdown('<span class="section-label">Residual calibration error (SSE)</span>', unsafe_allow_html=True)
    fig_e = go.Figure(go.Scatter(
        x=hw_calib_sorted["TradeDate"], y=hw_calib_sorted["ResidualError"],
        mode="lines", line=dict(color="#1E90FF", width=1),
        hovertemplate="%{x|%d %b %Y}: %{y:.2e}<extra></extra>"
    ))
    fig_e.add_vline(x=selected_date_hw, line=dict(color="#9ca3af", width=1, dash="dot"))
    fig_e.update_layout(
        height=200, margin=dict(l=0, r=0, t=8, b=0),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Inter", size=10),
        xaxis=dict(showgrid=True, gridcolor="#f3f4f6", showline=True, linecolor="#e5e7eb"),
        yaxis=dict(type="log", showgrid=True, gridcolor="#f3f4f6", showline=True, linecolor="#e5e7eb"),
    )
    st.plotly_chart(fig_e, use_container_width=True, config={"displayModeBar": False})

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MARKET CONVENTIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab_conv:
    st.markdown("<div style='margin-top:0.5rem;'></div>", unsafe_allow_html=True)

    def render_leg(leg_data: dict, title: str):
        rows = "".join(
            f"<div class='conv-row'><span class='conv-key'>{k}</span><span class='conv-val'>{v}</span></div>"
            for k, v in leg_data.items()
        )
        return f"<div class='conv-card'><div class='conv-title'>{title}</div>{rows}</div>"

    def render_overview(conv: dict, name: str):
        overview_keys = ["curve", "description", "settlement", "discounting", "is_end_end"]
        overview = {k: conv[k] for k in overview_keys if k in conv}
        rows = "".join(
            f"<div class='conv-row'><span class='conv-key'>{k}</span><span class='conv-val'>{str(v)}</span></div>"
            for k, v in overview.items()
        )
        return f"<div class='conv-card'><div class='conv-title'>{name}</div>{rows}</div>"

    col_estr, col_euribor = st.columns(2, gap="large")

    for col, curve_name in [(col_estr, "ESTR"), (col_euribor, "EURIBOR6M")]:
        with col:
            st.markdown(f'<span class="section-label">{curve_name}</span>', unsafe_allow_html=True)

            if curve_name not in conventions:
                st.warning(f"market_conventions/{curve_name.lower()}.json not found")
                continue

            conv = conventions[curve_name]

            st.markdown(render_overview(conv, "Overview"), unsafe_allow_html=True)

            if "fixed_leg" in conv:
                st.markdown(render_leg(conv["fixed_leg"], "Fixed leg"), unsafe_allow_html=True)

            if "float_leg" in conv:
                st.markdown(render_leg(conv["float_leg"], "Floating leg"), unsafe_allow_html=True)

    # Instruments table
    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown('<span class="section-label">Instrument tickers</span>', unsafe_allow_html=True)

    inst_col1, inst_col2 = st.columns(2, gap="large")

    for col, curve_name in [(inst_col1, "ESTR"), (inst_col2, "EURIBOR6M")]:
        with col:
            if curve_name not in conventions:
                continue
            conv = conventions[curve_name]
            if "instruments" not in conv:
                continue
            instruments_df = pd.DataFrame(conv["instruments"])
            st.markdown(f'<span class="section-label">{curve_name} — {len(instruments_df)} instruments</span>',
                        unsafe_allow_html=True)
            st.dataframe(instruments_df, use_container_width=True, height=400)

    # Vol surface conventions
    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown('<span class="section-label">Cap vol surface</span>', unsafe_allow_html=True)

    if "VOL_SURFACE" in conventions:
        vs = conventions["VOL_SURFACE"]

        vs_col1, vs_col2 = st.columns(2, gap="large")

        with vs_col1:
            # Overview card
            overview_keys = ["surface", "vol_unit", "underlying_index", "discounting",
                             "settlement", "reset_freq", "day_count"]
            overview_rows = "".join(
                f"<div class='conv-row'><span class='conv-key'>{k}</span><span class='conv-val'>{vs.get(k, '—')}</span></div>"
                for k in overview_keys
            )
            dr = vs.get("date_range", {})
            date_rows = "".join(
                f"<div class='conv-row'><span class='conv-key'>{k}</span><span class='conv-val'>{dr.get(k, '—')}</span></div>"
                for k in ["start", "end", "frequency", "n_obs"]
            )
            st.markdown(f"""
            <div class='conv-card'>
              <div class='conv-title'>Overview</div>
              {overview_rows}
            </div>
            <div class='conv-card'>
              <div class='conv-title'>Date range</div>
              {date_rows}
            </div>
            """, unsafe_allow_html=True)

            tenors_in    = vs.get("tenors_years_in_scope", [])
            tenors_out   = vs.get("tenors_years_out_of_scope", [])
            st.markdown(f"""
            <div class='conv-card'>
              <div class='conv-title'>Tenors in scope ({len(tenors_in)})</div>
              <div style='font-family: JetBrains Mono, monospace; font-size: 0.72rem; color: #374151; line-height: 2;'>
                {" · ".join(f"{t}Y" for t in tenors_in)}
              </div>
            </div>
            <div class='conv-card'>
              <div class='conv-title'>Tenors out of scope ({len(tenors_out)})</div>
              <div style='font-family: JetBrains Mono, monospace; font-size: 0.72rem; color: #9ca3af; line-height: 2;'>
                {" · ".join(f"{t}Y" for t in tenors_out) if tenors_out else "—"}
              </div>
            </div>
            """, unsafe_allow_html=True)

        with vs_col2:
            # Strikes table
            strikes = vs.get("strikes", [])
            strike_rows = ""
            for s in strikes:
                label     = s["strike_label"]
                stype     = s["strike_type"]
                val       = f"{s['strike_value_pct']:+.3f}%" if s["strike_value_pct"] is not None else "date-dependent"
                n_tickers = len(s.get("tenors", {}))
                strike_rows += (
                    f"<div class=\"conv-row\">"
                    f"<span class=\"conv-key\">{label}</span>"
                    f"<span class=\"conv-val\">{val}&nbsp;"
                    f"<span style=\"color:#9ca3af;font-weight:400\">({stype}, {n_tickers} tickers)</span>"
                    f"</span></div>"
                )

            st.markdown(
                f"<div class=\"conv-card\"><div class=\"conv-title\">Strikes ({len(strikes)})</div>{strike_rows}</div>",
                unsafe_allow_html=True
            )

            # Ticker table for selected strike
            strike_labels_list = [s["strike_label"] for s in strikes]
            selected_strike_label = st.selectbox(
                "Strike tickers", strike_labels_list,
                label_visibility="collapsed", key="strike_ticker_sel"
            )
            selected_strike = next(s for s in strikes if s["strike_label"] == selected_strike_label)
            ticker_rows = [
                (tenor, info.get("ticker", "—"), info.get("index", "—"), info.get("in_scope", False))
                for tenor, info in selected_strike.get("tenors", {}).items()
            ]
            ticker_df = pd.DataFrame(ticker_rows, columns=["Tenor", "Bloomberg ticker", "Index", "In scope"])
            st.dataframe(ticker_df, use_container_width=True, height=340)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="margin-top:2rem; padding-top:0.75rem; border-top:0.5px solid #e5e7eb;
     font-size:0.62rem; color:#9ca3af; font-family:'JetBrains Mono',monospace;
     display:flex; justify-content:space-between;">
  <span>Bocconi University · MAFINRISK · Supervisor: Prof. Rotondi</span>
  <span>data: Bloomberg BGN · curves: bootstrapped daily · vols: normal (Bachelier) bp · {len(all_dates)} business days</span>
</div>
""", unsafe_allow_html=True)
