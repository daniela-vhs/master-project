import streamlit as st
from helpers import get_trades, chart_layout, write_subtitle, write_subsubtitle, get_risk_table, date_slider, print_trade
from quant.dates import Tenor
import plotly.graph_objects as go
import numpy as np
import pandas as pd

def pnl_tab(actual_pnl_df, sens_df):
    actual_pnl_df = actual_pnl_df[actual_pnl_df.ValueDate > actual_pnl_df.TradeDate]

    sens_trades = get_trades(sens_df)
    
    pnl_trades = actual_pnl_df.drop_duplicates(
        subset = [
            "TradeDate",
            "Instrument",
            "TradeTenor",
            "TradeMaturity",
            "Strike",
        ]
    )[[
        "TradeDate",
        "Instrument",
        "TradeTenor",
        "TradeMaturity",
        "IsATM",
        "Strike",
        "Notional",
        "Position"
    ]].rename(
        {"Strike": "TradeStrike"},
        axis = 1
    ).to_dict("records")

    common_trades = set([tuple(i.items()) for i in sens_trades]) & set([tuple(i.items()) for i in pnl_trades])
    common_trades = sorted([dict(i) for i in common_trades], key = lambda x: (x["TradeDate"], x["TradeMaturity"]))

    pnl_trade = st.selectbox("Select trade:", ["Full portfolio"] + common_trades, index = 0, format_func = lambda x: x if x == "Full portfolio" else print_trade(x))

    sens_measures = st.multiselect(
        "Sens measures included:",
        ["Delta", "Gamma", "Vega", "Volga", "Vanna", "Theta"],
        ["Delta", "Gamma", "Vega", "Volga", "Vanna", "Theta"]
    )

    measure_order = {i[1]: i[0] for i in enumerate(["Delta", "Gamma", "Vega", "Volga", "Vanna", "Theta"])}

    pnl_components = st.multiselect(
        "PnL components included:",
        ["Rate", "Vol", "Time", "Cross", "Realized"],
        ["Rate", "Vol", "Time", "Cross", "Realized"],
    )

    pnl_order      = {i[1]: i[0] for i in enumerate(["Rate", "Vol", "Time", "Cross", "Realized"])}
    pnl_components = ["Rate", "Vol", "Time", "Cross", "Realized"] if len(pnl_components) == 0 else pnl_components
    pnl_components = sorted(pnl_components, key = lambda x: pnl_order[x])

    st.subheader("Explained vs residual")

    pnl_df = actual_pnl_df.copy() if pnl_trade == "Full portfolio" else\
        actual_pnl_df[
            (actual_pnl_df.Instrument == pnl_trade["Instrument"]) &
            (actual_pnl_df.TradeDate == pnl_trade["TradeDate"]) &
            (actual_pnl_df.TradeTenor == pnl_trade["TradeTenor"])
        ]

    sens_df = sens_df.copy() if pnl_trade == "Full portfolio" else\
        sens_df[
            (sens_df.TradeDate == pnl_trade["TradeDate"]) &
            (sens_df.TradeTenor == pnl_trade["TradeTenor"])
        ]

    pnl_date = date_slider(sorted(pnl_df.ValueDate.unique()), label = "Valuation date", key = "pnl_date")

    sens_df = sens_df[sens_df.Measure.isin(sens_measures)] if len(sens_measures) > 0 else sens_df

    pnl_pivot = pnl_df.pivot_table(index = "ValueDate", values = [f"{i}PnL" for i in pnl_components], aggfunc="sum").fillna(0)[[f"{i}PnL" for i in pnl_components]]

    if "RealizedPnL" in pnl_pivot:
        pnl_pivot["TotalPnL"] = pnl_pivot.drop("RealizedPnL", axis=1).sum(axis=1)
    else:
        pnl_pivot["TotalPnL"] = pnl_pivot.sum(axis=1)

    sens_pivot = sens_df.pivot_table(index = "ValueDate", columns = ["Model", "Measure"], values = "PnL", aggfunc="sum").fillna(0)

    day_pnl = pnl_pivot.loc[pnl_date].TotalPnL.sum()

    col1, col2 = st.columns(2)

    for n, col in enumerate([col1, col2]):
        model = ["FMM", "HW"][n]
        model_name = model.replace("HW", "Hull-White")
        key = f"pnl-{model}"

        with col:
            write_subsubtitle(model_name)

            components = {k: v for k, v in sens_pivot[model].loc[pnl_date].to_dict().items() if v != 0}
            components = dict(sorted(components.items(), key = lambda x: measure_order[x[0]]))

            sens_pnl   = sum(components.values())
            residual   = day_pnl - sens_pnl

            names  = list(components.keys()) + ["Explained", "Residual", "Actual"]
            values = list(components.values()) + [sens_pnl] + [residual] + [day_pnl]
            types  = ["relative" for i in components.values()] + ["total", "relative", "total"]
            text   = [f"{i:0,.0f}" if n != len(values) - 2 else f"{abs(1 - sens_pnl / day_pnl):.0%}" for n, i in enumerate(values)]

            fig = go.Figure()

            fig.add_trace(
                go.Waterfall(
                    name = model,
                    orientation = "v",
                    measure = types,
                    x = names,
                    y = values,
                    text = text,
                    decreasing = dict(
                        marker = dict(
                            color = "crimson",
                        )
                    ),
                    increasing = dict(
                        marker = dict(
                            color = "steelblue",
                        )
                    ),
                    totals = dict(
                        marker = dict(
                            color = "#444",
                        )
                    )
                )
            )

            fig.update_layout(
                **chart_layout(),
            )

            fig.update_yaxes(
                title = dict(
                    text = "PnL Contribution",
                    font = dict(
                        size = 12,
                    )
                ),
            )

            fig.update_xaxes(
                tickangle = -90
            )

            st.plotly_chart(fig, key=key, height=300)

    st.caption(
        "Categories with negligible contribution are omitted. "
        "Hull-White's Vega, Volga, and Vanna are not shown when their "
        "recalibrated contribution on the selected date is exactly zero."
    )

    st.caption(
        r"Residual label: $\Big|1 - \frac{\text{ExplainedPnL}}{\text{ActualPnL}}\Big|$, non-explained portion of Actual PnL."
    )

    st.divider()

    st.subheader("Historical residuals")

    residual_group = st.radio(
        "Group by:",
        ["Week", "Month", "Year"],
        index = 1,
        horizontal = True,
        key = "residual_group"
    )

    residual_group = {"Day": "D", "Week": "W", "Month": "ME", "Year": "YE"}[residual_group]

    joint_pivot      = pnl_pivot[["TotalPnL"]].join(sens_pivot.T.groupby("Model").sum().T)
    joint_pivot.FMM -= joint_pivot.TotalPnL
    joint_pivot.HW  -= joint_pivot.TotalPnL
    joint_pivot      = abs(joint_pivot).resample(residual_group).mean().dropna()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x = joint_pivot.index,
            y = joint_pivot.FMM,
            name = "FMM residual",
            mode = "lines+markers" if residual_group != "W" else "lines",
            line = dict(
                color = "dodgerblue",
                shape = "spline",
                width = 1.2,
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x = joint_pivot.index,
            y = joint_pivot.HW,
            name = "Hull-White residual",
            mode = "lines+markers" if residual_group != "W" else "lines",
            line = dict(
                color = "tomato",
                shape = "spline",
                width = 1.2,
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x = joint_pivot.index,
            y = joint_pivot.TotalPnL,
            name = "Actual PnL (mean |daily|, scale reference)",
            mode = "lines+markers" if residual_group != "W" else "lines",
            line = dict(
                dash = "dot",
                color = "black",
                shape = "spline",
                width = 1.2,
            ),
        )
    )

    fig.update_layout(
        **chart_layout(),
        yaxis = dict(
            type = "log",
            title = "Mean daily |PnL| (log scale)",
        ),
    )

    st.plotly_chart(fig, height=300)

    st.divider()

    st.subheader("Actual PnL evolution")

    pnl_group = st.radio(
        "Group by:",
        ["Week", "Month", "Year"],
        index = 1,
        horizontal = True,
        key = "pnl_group"
    )

    pnl_group = {"Day": "D", "Week": "W", "Month": "ME", "Year": "YE"}[pnl_group]

    grouped_pnl = pnl_pivot.resample(pnl_group).sum().dropna()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x = grouped_pnl.index,
            y = grouped_pnl.TotalPnL,
            name = "Total PnL",
            mode = "lines+markers",
            line = dict(
                color = "black",
                dash = "dot"
            )
        )
    )

    fig.add_trace(
        go.Bar(
            x = grouped_pnl.index,
            y = grouped_pnl.RatePnL,
            name = "Rate PnL"
        )
    )

    fig.add_trace(
        go.Bar(
            x = grouped_pnl.index,
            y = grouped_pnl.VolPnL,
            name = "Rate PnL"
        )
    )

    fig.add_trace(
        go.Bar(
            x = grouped_pnl.index,
            y = grouped_pnl.TimePnL,
            name = "Time PnL"
        )
    )

    fig.add_trace(
        go.Bar(
            x = grouped_pnl.index,
            y = grouped_pnl.CrossPnL,
            name = "Cross PnL"
        )
    )

    fig.add_trace(
        go.Bar(
            x = grouped_pnl.index,
            y = grouped_pnl.RealizedPnL,
            name = "Realized PnL"
        )
    )

    fig.update_layout(
        **chart_layout(),
        barmode = "relative",
    )

    st.plotly_chart(fig, height = 300)