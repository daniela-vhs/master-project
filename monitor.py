import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
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
.block-container { padding: 2.5rem 2rem 1.5rem; max-width: 1600px; }
header[data-testid="stHeader"] { background: transparent; }
div[data-testid="stDecoration"] { display: none; }
.thesis-header { border-bottom: 1.5px solid #1a1a2e; padding-bottom: 0.75rem; margin-bottom: 1.25rem; }
.thesis-title  { font-size: 1.05rem; font-weight: 600; color: #1a1a2e; letter-spacing: -0.01em; margin: 0; }
.thesis-sub    { font-size: 0.72rem; color: #6b7280; font-family: 'JetBrains Mono', monospace; margin: 0.15rem 0 0; }
.section-label { font-size: 0.62rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #9ca3af; margin-bottom: 0.4rem; display: block; }
.pill { display: inline-block; background: #f0f4ff; color: #1e3a8a; border-radius: 4px; padding: 2px 8px; font-size: 0.68rem; font-family: 'JetBrains Mono', monospace; font-weight: 500; margin-right: 6px; }
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
def load_rates():
    df = pd.read_parquet("clean_data/rates.parquet")
    df["Date"] = pd.to_datetime(df["Date"])
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

zero_rates  = load_zero_rates()
vols        = load_vols()
rates_df    = load_rates()
conventions = load_conventions()

all_dates   = sorted(zero_rates["TradeDate"].unique())

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="thesis-header">
  <p class="thesis-title">Greeks & PnL Explained: Hull-White vs FMM</p>
  <p class="thesis-sub">EUR cap/caplet pricing · Euribor 6M · ESTR OIS discounting · Normal (Bachelier) vols</p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_curves, tab_vols, tab_conv = st.tabs(["Zero curves", "Vol surface", "Market conventions"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ZERO CURVES
# ══════════════════════════════════════════════════════════════════════════════
with tab_curves:
    date_idx = st.slider(
        "Trade date",
        min_value=0, max_value=len(all_dates) - 1,
        value=int(len(all_dates) * 0.6),
        format="", label_visibility="collapsed",
        key="date_slider_curves"
    )
    selected_date = pd.Timestamp(all_dates[date_idx])

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        st.markdown(f'<span class="pill">📅 {selected_date.strftime("%d %b %Y")}</span>'
                    f'<span class="pill">day {date_idx + 1} / {len(all_dates)}</span>',
                    unsafe_allow_html=True)

    st.markdown("<div style='margin:0.75rem 0;'></div>", unsafe_allow_html=True)

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<span class="section-label">Curve selector</span>', unsafe_allow_html=True)
        curve_choice = st.radio("Curve", ["ESTR", "EURIBOR6M", "Both"],
                                horizontal=True, label_visibility="collapsed")

        day_data = zero_rates[zero_rates["TradeDate"] == selected_date].copy()
        day_data["DaysOffset"] = (day_data["Maturity"] - selected_date).dt.days
        day_data["ZeroRate"]   = np.where(
            day_data["DaysOffset"] > 0,
            -np.log(day_data["DiscountFactor"]) / (day_data["DaysOffset"] / 365),
            np.nan
        )

        fig = go.Figure()
        colors = {"ESTR": "#1e3a8a", "EURIBOR6M": "#b45309"}
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
            mode="lines", line=dict(color=colors.get(hist_curve, "#1e3a8a"), width=1.5),
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

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — VOL SURFACE
# ══════════════════════════════════════════════════════════════════════════════
with tab_vols:
    vol_dates     = sorted(vols["Date"].unique())
    date_idx_v    = st.slider(
        "Trade date",
        min_value=0, max_value=len(vol_dates) - 1,
        value=int(len(vol_dates) * 0.6),
        format="", label_visibility="collapsed",
        key="date_slider_vols"
    )
    selected_date_v = pd.Timestamp(vol_dates[date_idx_v])
    st.markdown(f'<span class="pill">📅 {selected_date_v.strftime("%d %b %Y")}</span>',
                unsafe_allow_html=True)
    st.markdown("<div style='margin:0.75rem 0;'></div>", unsafe_allow_html=True)

    day_vols = vols[vols["Date"] == selected_date_v].copy()

    # ATM rates from par rates (Euribor 6M swap rates as proxy for ATM forward)
    atm_rates_raw = rates_df[
        (rates_df["Date"] == selected_date_v) &
        (rates_df["Curve"] == "EURIBOR6M")
    ].set_index("Tenor")["Rate"]

    tenor_to_swap = {
        "3Y": "3Y", "4Y": "4Y", "5Y": "5Y", "6Y": "6Y", "7Y": "7Y",
        "8Y": "8Y", "9Y": "9Y", "10Y": "10Y", "12Y": "12Y",
    }
    cap_tenors     = list(tenor_to_swap.keys())
    otm_vols       = day_vols[~day_vols["IsATM"]].copy()
    atm_vols       = day_vols[day_vols["IsATM"]].copy()
    strikes_sorted = sorted(otm_vols["Strike"].dropna().unique())
    strike_labels  = [f"{s:+.3f}%" if s != 0.0 else "0.000%" for s in strikes_sorted]

    # ── Vol table ─────────────────────────────────────────────────────────────
    left_v, right_v = st.columns([3, 2], gap="large")

    with left_v:
        st.markdown('<span class="section-label">Cap normal vol surface — bp (Bachelier)</span>', unsafe_allow_html=True)

        header = "<tr><th class='tc'>Tenor</th><th style='text-align:center;'>ATM rate</th><th class='atm-col'>ATM vol</th>"
        for sl in strike_labels:
            header += f"<th>{sl}</th>"
        header += "</tr>"

        rows = ""
        for tenor in cap_tenors:
            swap_tenor = tenor_to_swap[tenor]
            atm_rate   = atm_rates_raw.get(swap_tenor, None)
            atm_rate_s = f"{atm_rate:.3f}%" if atm_rate is not None else "—"

            atm_row = atm_vols[atm_vols["Tenor"] == tenor]
            atm_vol = f"{atm_row['Vol'].iloc[0]:.1f}" if not atm_row.empty else "—"

            row = f"<tr><td class='tc'>{tenor}</td><td style='text-align:center;'><span class='atm-rate'>{atm_rate_s}</span></td><td class='atm-col'>{atm_vol}</td>"
            for s in strikes_sorted:
                cell = otm_vols[(otm_vols["Tenor"] == tenor) & (otm_vols["Strike"] == s)]
                val  = f"{cell['Vol'].iloc[0]:.1f}" if not cell.empty else "—"
                row += f"<td>{val}</td>"
            row += "</tr>"
            rows += row

        st.markdown(f"""
        <div style="overflow-x:auto; max-height:460px; overflow-y:auto; margin-bottom:1rem;">
        <table class="vol-table">
          <thead style="position:sticky;top:0;background:white;z-index:1;">
            {header}
            <tr><td colspan="{3+len(strikes_sorted)}" style="height:1px;background:#e5e7eb;padding:0;"></td></tr>
          </thead>
          <tbody>{rows}</tbody>
        </table>
        </div>
        """, unsafe_allow_html=True)

    with right_v:
        # ── Heatmap ───────────────────────────────────────────────────────────
        st.markdown('<span class="section-label">Vol surface heatmap</span>', unsafe_allow_html=True)

        z_rows = []
        for tenor in cap_tenors:
            row_vals = []
            for s in strikes_sorted:
                cell = otm_vols[(otm_vols["Tenor"] == tenor) & (otm_vols["Strike"] == s)]
                row_vals.append(float(cell["Vol"].iloc[0]) if not cell.empty else np.nan)
            z_rows.append(row_vals)

        z_text = [[f"{v:.1f}" if not np.isnan(v) else "" for v in row] for row in z_rows]

        fig2 = go.Figure(go.Heatmap(
            z=z_rows,
            x=strike_labels,
            y=cap_tenors,
            colorscale=[[0, "#dbeafe"], [0.5, "#3b82f6"], [1, "#1e3a8a"]],
            text=z_text,
            texttemplate="%{text}",
            textfont=dict(size=9, family="JetBrains Mono"),
            hovertemplate="Tenor: %{y}<br>Strike: %{x}<br>Vol: %{z:.1f} bp<extra></extra>",
            showscale=True,
            colorbar=dict(title=dict(text="bp", font=dict(size=10)), thickness=10, len=0.9),
            xgap=1, ygap=1
        ))
        fig2.update_layout(
            height=340, margin=dict(l=0, r=40, t=8, b=0),
            paper_bgcolor="white", plot_bgcolor="white",
            font=dict(family="Inter", size=10),
            xaxis=dict(
                title=dict(text="Strike", font=dict(size=10, color="#9ca3af")),
                tickangle=-45,
                type="category",
                tickmode="array",
                tickvals=strike_labels,
                ticktext=strike_labels,
            ),
            yaxis=dict(
                title=dict(text="Cap tenor", font=dict(size=10, color="#9ca3af")),
                autorange="reversed",
                type="category"
            )
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

        # ── ATM vol time series ───────────────────────────────────────────────
        st.markdown('<span class="section-label">ATM vol — historical</span>', unsafe_allow_html=True)
        atm_tenor_opt = st.selectbox("Cap tenor", cap_tenors, index=4,
                                     label_visibility="collapsed", key="atm_tenor")
        hist_atm = vols[(vols["IsATM"]) & (vols["Tenor"] == atm_tenor_opt)].copy()
        fig4 = go.Figure(go.Scatter(
            x=hist_atm["Date"], y=hist_atm["Vol"],
            mode="lines", line=dict(color="#1e3a8a", width=1.5),
            hovertemplate="%{x|%d %b %Y}: %{y:.1f} bp<extra></extra>"
        ))
        fig4.update_layout(
            height=200, margin=dict(l=0, r=0, t=8, b=0),
            paper_bgcolor="white", plot_bgcolor="white",
            font=dict(family="Inter", size=10),
            xaxis=dict(showgrid=True, gridcolor="#f3f4f6", showline=True, linecolor="#e5e7eb"),
            yaxis=dict(showgrid=True, gridcolor="#f3f4f6", ticksuffix=" bp",
                       title=dict(text="Normal vol", font=dict(size=10, color="#9ca3af")),
                       showline=True, linecolor="#e5e7eb"),
        )
        st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MARKET CONVENTIONS
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

            tenors = vs.get("tenors_years", [])
            st.markdown(f"""
            <div class='conv-card'>
              <div class='conv-title'>Tenors ({len(tenors)})</div>
              <div style='font-family: JetBrains Mono, monospace; font-size: 0.72rem; color: #374151; line-height: 2;'>
                {" · ".join(f"{t}Y" for t in tenors)}
              </div>
            </div>
            """, unsafe_allow_html=True)

        with vs_col2:
            # Strikes table
            strikes = vs.get("strikes", [])
            strike_rows = ""
            for s in strikes:
                label = s["strike_label"]
                stype = s["strike_type"]
                val   = f"{s['strike_value_pct']:+.3f}%" if s["strike_value_pct"] is not None else "date-dependent"
                n_tickers = len(s.get("tickers", {}))
                strike_rows += f"""
                <div class='conv-row'>
                  <span class='conv-key'>{label}</span>
                  <span class='conv-val'>{val} &nbsp;<span style='color:#9ca3af;font-weight:400;'>({stype}, {n_tickers} tickers)</span></span>
                </div>"""

            st.markdown(f"""
            <div class='conv-card'>
              <div class='conv-title'>Strikes ({len(strikes)})</div>
              {strike_rows}
            </div>
            """, unsafe_allow_html=True)

            # Ticker table for selected strike
            strike_labels_list = [s["strike_label"] for s in strikes]
            selected_strike_label = st.selectbox(
                "Strike tickers", strike_labels_list,
                label_visibility="collapsed", key="strike_ticker_sel"
            )
            selected_strike = next(s for s in strikes if s["strike_label"] == selected_strike_label)
            ticker_df = pd.DataFrame(
                [(k, v) for k, v in selected_strike["tickers"].items()],
                columns=["Tenor", "Bloomberg ticker"]
            )
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
