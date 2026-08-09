import streamlit as st
from helpers import get_trades, chart_layout, write_subtitle, write_subsubtitle, get_risk_table, date_slider
from dates import Tenor
import plotly.graph_objects as go
import numpy as np

def pnl_tab(actual_pnl_df, sens_df):
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

    common_trades = sorted(
        [dict(i) for i in common_trades],
        key = lambda x: (x["TradeDate"], Tenor(x["TradeTenor"]))
    )

    st.subheader("Actual PnL vs Explained PnL")

    pnl_trade = st.selectbox(
        "Select trade:",
        common_trades,
        key = "pnl_trade",
        format_func = lambda x: f'{x["Instrument"]}({x["TradeDate"].date()}, {x["TradeTenor"]})'
    )

    pnl_df = actual_pnl_df[
        (actual_pnl_df.TradeDate == pnl_trade["TradeDate"]) &
        (actual_pnl_df.Instrument == pnl_trade["Instrument"]) &
        (actual_pnl_df.TradeTenor == pnl_trade["TradeTenor"]) &
        (actual_pnl_df.TradeMaturity == pnl_trade["TradeMaturity"]) &
        (actual_pnl_df.Strike == pnl_trade["TradeStrike"]) &
        (actual_pnl_df.ValueDate > actual_pnl_df.TradeDate)
    ].set_index("ValueDate")

    sens_pivot = get_risk_table(
        pnl_trade,
        sens_df
    ).rename(
        {"ValueDate": "SensDate"},
        axis=1
    ).set_index("SensDate").join(
        pnl_df.reset_index().set_index("PrevDate")[["ValueDate"]]
    ).reset_index().pivot_table(
        index = ["ValueDate", "Measure"],
        columns = "Model",
        values = "PnL",
        aggfunc = "sum"
    )

    # ------------------------------------- #
    #         ACTUAL PNL EVOLUTION          #
    # ------------------------------------- #
    st.subheader("Explained vs Residual")
    value_date = date_slider(sens_pivot.index.get_level_values("ValueDate"), key = "pnl_value_date")

    sens_measures = st.multiselect(
        "Measures included",
        ["Delta", "Gamma", "Vega", "Volga", "Vanna", "Theta"],
        ["Delta", "Gamma", "Vega", "Volga", "Vanna", "Theta"],
    )

    day_pnl = pnl_df.loc[value_date].TotalPnL

    col1, col2 = st.columns(2, gap="large")

    for n, col in enumerate([col1, col2]):
        model = ["FMM", "HW"][n]
        with col:
            values = {
                i: sens_pivot.loc[value_date].fillna(0).loc[i][model] for i in sens_measures
            }

            residual = pnl_df.loc[value_date][["TotalPnL"]].sum() - sum(list(values.values()))

            fig = go.Figure()

            valid_values = {key: value for key, value in values.items() if value != 0}

            fig.add_trace(
                go.Waterfall(
                    name = "20",
                    orientation = "v",
                    measure = ["relative" for i in list(valid_values.keys())] + ["total", "relative", "total"],
                    x = list(valid_values.keys()) + ["Explained", "Residual", "Actual"],
                    y = list(valid_values.values()) + [sum(list(valid_values.values())), residual, day_pnl],
                    text = [f"{i:.0f}" for i in list(valid_values.values()) + [sum(list(valid_values.values())), residual, day_pnl]],
                    decreasing = dict(
                        marker = dict(
                            color = "tomato",
                        )
                    ),
                    increasing = dict(
                        marker = dict(
                            color = "dodgerblue",
                        )
                    ),
                    totals = dict(
                        marker = dict(
                            color = "darkslateblue",
                        )
                    ),
                )
            )

            fig.update_layout(
                **chart_layout()
            )

            fig.update_yaxes(
                title = dict(
                    text = "PnL Contribution",
                    font = dict(
                        size = 12,
                    )
                )
            )

            st.plotly_chart(fig, height=300, key=f"pnl_explained_{model}")

    # ------------------------------------- #
    #          HISTORICAL RESIDUAS          #
    # ------------------------------------- #

    st.subheader("Historical residuals")

    col1, col2, col3 = st.columns(3)

    with col1:
        sens_order = st.radio(
            "Measure order",
            ["First", "Second", "Cross", "Total"],
            index = 3,
            horizontal = True,
            key = "sens_order",
        )

    with col2:
        pnl_component = st.radio(
            "PnL component",
            ["Rate", "Vol", "Cross", "Time", "Total"],
            index = 4,
            horizontal = True,
            key = "pnl_component"
        )

    with col3:
        pnl_group_by = st.radio(
            "Group by",
            ["Day", "Week", "Month", "Year"],
            index = 1,
            key = "pnl_group_by",
            horizontal = True,
        )

    pnl_group_by = {"Day": "D", "Week": "W", "Month": "ME", "Year": "YE"}[pnl_group_by]

    sens_measures = ["Delta", "Gamma", "Vega", "Volga", "Vanna", "Theta"] if sens_order == "Total" else ["Delta", "Vega", "Theta"] if sens_order == "First" else ["Gamma", "Volga"] if sens_order == "Second" else ["Vanna"]

    pnl_measures = ["Delta", "Gamma", "Vega", "Volga", "Vanna", "Theta"] if pnl_component == "Total" else ["Delta", "Gamma"] if pnl_component == "Rate" else ["Vega", "Volga"] if pnl_component == "Vol" else ["Vanna"] if pnl_component == "Cross" else ["Theta"]

    selected_measures = set(sens_measures) & set(pnl_measures)

    pnl_columns = [f"{pnl_component}PnL"]

    if len(selected_measures) == 0:
        if pnl_component == "Time":
            selected_measures = {"Theta"}

        elif pnl_component == "Cross" or sens_order == "Cross":
            selected_measures = {"Vanna"}

    fig = go.Figure()

    for m, model in enumerate(["FMM", "HW"]):
        name = model.replace("HW", "Hull-White")
        color = ["dodgerblue", "tomato"][m]

        hist_df = sens_pivot[[model]]
        hist_df = hist_df[hist_df.index.get_level_values("Measure").isin(selected_measures)]

        hist_df = hist_df.groupby("ValueDate").sum().join(pnl_df[pnl_columns]).rename({
            pnl_columns[0]: "ActualPnL",
            model: "ModelPnL",
        }, axis=1)

        hist_df["Residual"] = abs(hist_df.ActualPnL - hist_df.ModelPnL)

        hist_df = hist_df.fillna(0).resample(pnl_group_by).sum() if pnl_group_by == "D" else hist_df.fillna(0).resample(pnl_group_by).mean()

        try:
            min_y_pnl = min(min_y_pnl, hist_df.Residual.min())
        except:
            min_y_pnl = hist_df.Residual.min()

        try:
            max_y_pnl = max(max_y_pnl, hist_df.Residual.max())
        except:
            max_y_pnl = hist_df.Residual.max()

        fig.add_trace(
            go.Scatter(
                x = hist_df.index,
                y = hist_df.Residual,
                name = name,
                mode = "lines+markers" if pnl_group_by != "D" else "lines",
                line = dict(
                    color = color,
                    shape = "spline",
                    smoothing = 1.3
                ),
                marker = dict(
                    size = 6,
                )
            )
        )

    fig.add_trace(
        go.Scatter(
            x = np.repeat(value_date, 2),
            y = [min_y_pnl, max_y_pnl],
            mode = "lines",
            line = dict(
                color = "slateblue",
                dash = "dot",
            )
        )
    )

    fig.update_layout(
        **chart_layout()
    )

    st.plotly_chart(fig, height = 300)

    # ------------------------------------- #
    #         ACTUAL PNL EVOLUTION          #
    # ------------------------------------- #

    st.subheader("Actual PnL evolution")

    col1, col2 = st.columns(2)

    with col1:
        pnl_type = st.radio(
            "PnL type:",
            ["Total", "Realized", "Split"],
            key = "pnl_type",
            horizontal = True,
        )

    with col2:
        group_pnl = st.radio(
            "Group by:",
            ["Day", "Week", "Month", "Year"],
            index = 1,
            key = "group_pnl",
            horizontal = True,
        )

        group_pnl = {"Day": "D", "Week": "W", "Month": "ME", "Year": "YE"}[group_pnl]

    fig = go.Figure()

    if pnl_type == "Split":
        max_y = max(
            np.maximum(0, pnl_df.resample(group_pnl).RatePnL.sum())\
            + np.maximum(0, pnl_df.resample(group_pnl).VolPnL.sum())\
            + np.maximum(0, pnl_df.resample(group_pnl).CrossPnL.sum())\
            + np.maximum(0, pnl_df.resample(group_pnl).TimePnL.sum())\
            + np.maximum(0, pnl_df.resample(group_pnl).RealizedPnL.sum())
        )
        
        min_y = min(
            np.minimum(0, pnl_df.resample(group_pnl).RatePnL.sum())\
            + np.minimum(0, pnl_df.resample(group_pnl).VolPnL.sum())\
            + np.minimum(0, pnl_df.resample(group_pnl).CrossPnL.sum())\
            + np.minimum(0, pnl_df.resample(group_pnl).TimePnL.sum())\
            + np.minimum(0, pnl_df.resample(group_pnl).RealizedPnL.sum())
        )

        fig.add_trace(
            go.Bar(
                x = pnl_df.resample(group_pnl).last().index,
                y = pnl_df.resample(group_pnl).RatePnL.sum(),
                name = "Rate PnL",
                marker = dict(
                    color = "dodgerblue"
                )
            )
        )

        fig.add_trace(
            go.Bar(
                x = pnl_df.resample(group_pnl).last().index,
                y = pnl_df.resample(group_pnl).VolPnL.sum(),
                name = "VolPnL",
                marker = dict(
                    color = "darkslateblue",
                )
            )
        )

        fig.add_trace(
            go.Bar(
                x = pnl_df.resample(group_pnl).last().index,
                y = pnl_df.resample(group_pnl).CrossPnL.sum(),
                name = "CrossPnL",
                marker = dict(
                    color = "crimson",
                )
            )
        )

        fig.add_trace(
            go.Bar(
                x = pnl_df.resample(group_pnl).last().index,
                y = pnl_df.resample(group_pnl).TimePnL.sum(),
                name = "TimePnL",
                marker = dict(
                    color = "silver",
                )
            )
        )

        fig.add_trace(
            go.Bar(
                x = pnl_df.resample(group_pnl).last().index,
                y = pnl_df.resample(group_pnl).RealizedPnL.sum(),
                name = "RealizedPnL",
                marker = dict(
                    color = "tomato"
                )
            )
        )

        if group_pnl != "D":
            fig.add_trace(
                go.Scatter(
                    x = pnl_df.resample(group_pnl).last().index,
                    y = (pnl_df.resample(group_pnl).TotalPnL.sum() + pnl_df.resample(group_pnl).RealizedPnL.sum()),
                    name = "TotalPnL",
                    mode = "lines+markers",
                    marker = dict(
                        color = "black",
                        line = dict(
                            color = "white",
                            width = 1.3
                        ),
                        size = 8
                        # size = 1.2
                    ),
                    line = dict (
                        shape = "spline",
                        smoothing = 0.25,
                        width = 1.3
                    )
                )
            )

    elif pnl_type == "Realized":
        min_y = pnl_df.resample(group_pnl).RealizedPnL.sum().min()
        max_y = pnl_df.resample(group_pnl).RealizedPnL.sum().max()
        fig.add_trace(
            go.Bar(
                x = pnl_df.resample(group_pnl).last().index,
                y = pnl_df.resample(group_pnl).RealizedPnL.sum(),
                name = "Realized PnL",
                marker = dict(
                    color = "tomato"
                )
            )
        )

    elif pnl_type == "Total":
        min_y = (pnl_df.resample(group_pnl).TotalPnL.sum() + pnl_df.resample(group_pnl).RealizedPnL.sum()).min()
        max_y = (pnl_df.resample(group_pnl).TotalPnL.sum() + pnl_df.resample(group_pnl).RealizedPnL.sum()).max()
        fig.add_trace(
            go.Bar(
                x = pnl_df.resample(group_pnl).last().index,
                y = (pnl_df.resample(group_pnl).TotalPnL.sum() + pnl_df.resample(group_pnl).RealizedPnL.sum()),
                name = "Realized PnL",
                marker = dict(
                    color = "dodgerblue",
                )
            )
        )

    fig.add_trace(
        go.Scatter(
            x = np.repeat(value_date, 2),
            y = [min_y, max_y],
            mode = "lines",
            line = dict(
                color = "slateblue",
                dash = "dot",
            ),
            name = "Value date",
        )
    )

    fig.update_yaxes(
        nticks = 10,
        title = dict(
            text = "PnL"
        )
    )

    fig.update_layout(
        barmode = "relative",
        **chart_layout(),
    )

    st.plotly_chart(fig, height=300)