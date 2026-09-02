from helpers import get_trades, print_trade, get_risk_table, sort_tenor, date_slider, apply_heatmap, write_subtitle, write_subsubtitle, chart_layout
from quant.dates import Tenor
import plotly.graph_objects as go
import streamlit as st
import numpy as np

def strike_cols(df):
    df.columns = [f"{i:.3%}" for i in df]
    return df

def risk_tab(sens_df):
    st.subheader("Greeks")

    trades = get_trades(sens_df)
    trades = sorted(trades, key=lambda x: (x["TradeDate"], Tenor(x["TradeTenor"])))

    selected_trade = st.selectbox(
        "Select trade:",
        trades,
        format_func = lambda x: print_trade(x), index=0
    )

    risk_table = get_risk_table(selected_trade, sens_df)

    rate_pivot = sort_tenor(
        risk_table.pivot_table(
            index = ["SensDate", "RateTenor"],
            columns = ["Measure", "Source", "Model"],
            values = "Value",
            aggfunc = "sum"
        ),
        "RateTenor"
    ) / 1_000_000
    
    vol_pivot = risk_table.pivot_table(
        index = ["SensDate", "VolTenor"],
        columns = ["Measure", "Strike", "Model"],
        values = "Value",
        aggfunc = "sum"
    ) / 1_000_000

    cross_pivot = risk_table.pivot_table(
        index = ["SensDate", "RateTenor", "VolTenor"],
        columns = ["Source", "Strike", "Model"],
        values = "Value",
        aggfunc = "sum"
    ) / 1_000_000

    # Valid dates for this trade
    value_date = date_slider(
        risk_table.SensDate.unique(),
        key = "pnl_date_slider",
        label = "Valuation date"
    )

    cmap = "RdBu"

    #############
    #   Delta   #
    #############

    st.subheader("Delta")

    write_subtitle(f"As of {value_date.date()}")

    col1, col2 = st.columns(2)

    for n, col in enumerate([col1, col2]):
        curve = ["Euribor 6M", "ESTR"][n]
        with col:
            write_subsubtitle(curve)
            st.dataframe(
                apply_heatmap(
                    rate_pivot["Delta"][curve.upper().replace(
                        " ",
                        ""
                    )].loc[value_date].dropna(
                        how = "all"
                    ).fillna(0).rename_axis(
                        "Tenor"
                    ).rename(
                        {"HW": "Hull-White"},
                        axis = 1
                    ),
                    cmap
                ),
                height = "content",
            )

    write_subtitle("Historical evolution")

    delta_tenors = sorted(
        rate_pivot["Delta"]["EURIBOR6M"].dropna()
        .index.get_level_values("RateTenor")\
        .unique(),
        key = lambda x: Tenor(x)
    )

    delta_tenor = st.selectbox(
        "Tenor",
        delta_tenors,
        index = 2,
        key = "delta_tenor"
    )

    col1, col2 = st.columns(2)

    delta_df = rate_pivot["Delta"].xs(
        delta_tenor,
        level="RateTenor"
    )

    for n, col in enumerate([col1, col2]):
        curve = ["Euribor 6M", "ESTR"][n]

        min_y = min(
            delta_df[curve.upper().replace(" ", "")].FMM.min(),
            delta_df[curve.upper().replace(" ", "")].HW.min()
        )

        max_y = min(
            delta_df[curve.upper().replace(" ", "")].FMM.max(),
            delta_df[curve.upper().replace(" ", "")].HW.max()
        )

        with col:
            write_subsubtitle(curve)

            fig = go.Figure()

            for m, model in enumerate(["FMM", "HW"]):
                color = ["dodgerblue", "tomato"][m]
                fig.add_trace(
                    go.Scatter(
                        x = delta_df[curve.upper().replace(" ", "")].index,
                        y = delta_df[curve.upper().replace(" ", "")][model],
                        name = model.replace("HW", "Hull-White"),
                        mode = "lines",
                        line = dict(
                            color = color,
                            width = 1.1,
                            shape = "spline",
                        )
                    )
                )

            fig.add_trace(
                go.Scatter(
                    x = np.repeat(value_date, 2),
                    y = [min_y, max_y],
                    mode = "lines",
                    name = "Value date",
                    line = dict(
                        color = "slateblue",
                        dash = "dot",
                    )
                )
            )

            fig.update_layout(
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
                ),
            )

            fig.update_yaxes(
                nticks = 5,
            )

            st.plotly_chart(fig, height=200)

    st.divider()

    #############
    #   Gamma   #
    #############

    st.subheader("Gamma")

    write_subtitle(f"As of {value_date.date()}")

    col1, col2 = st.columns(2)

    for n, col in enumerate([col1, col2]):
        curve = ["Euribor 6M", "ESTR"][n]

        with col:
            write_subsubtitle(curve)
            st.dataframe(
                apply_heatmap(
                    rate_pivot["Gamma"][curve.upper().replace(" ", "")].loc[value_date].dropna(
                        how = "all"
                    ).fillna(0).rename_axis(
                        "Tenor"
                    ).rename(
                        {"HW": "Hull-White"},
                        axis = 1
                    ),
                    cmap
                ),
                height = "content",
            )

    write_subtitle("Historical evolution")

    gamma_tenors = sorted(
        rate_pivot["Gamma"]["EURIBOR6M"].dropna()
        .index.get_level_values("RateTenor")\
        .unique(),
        key = lambda x: Tenor(x)
    )

    gamma_tenor = st.selectbox(
        "Tenor",
        gamma_tenors,
        index = 2,
        key = "gamma_tenor"
    )

    col1, col2 = st.columns(2)

    gamma_df = rate_pivot["Gamma"].xs(
        gamma_tenor,
        level="RateTenor"
    )

    for n, col in enumerate([col1, col2]):
        curve = ["Euribor 6M", "ESTR"][n]

        with col:
            write_subsubtitle(curve)

            fig = go.Figure()

            for m, model in enumerate(["FMM", "HW"]):
                color = ["dodgerblue", "tomato"][m]

                fig.add_trace(
                    go.Scatter(
                        x = gamma_df[curve.upper().replace(" ", "")].index,
                        y = gamma_df[curve.upper().replace(" ", "")][model],
                        name = model.replace("HW", "Hull-White"),
                        mode = "lines",
                        line = dict(
                            color = color,
                            width = 1.1,
                            shape = "spline",
                        )
                    )
                )

            min_y = min(
                gamma_df[curve.upper().replace(" ", "")].FMM.min(),
                gamma_df[curve.upper().replace(" ", "")].HW.min()
            )

            max_y = min(
                gamma_df[curve.upper().replace(" ", "")].FMM.max(),
                gamma_df[curve.upper().replace(" ", "")].HW.max()
            )

            fig.add_trace(
                go.Scatter(
                    x = np.repeat(value_date, 2),
                    y = [min_y, max_y],
                    mode = "lines",
                    name = "Value date",
                    line = dict(
                        color = "slateblue",
                        dash = "dot",
                    )
                )
            )

            fig.update_layout(
                **chart_layout()
            )

            fig.update_yaxes(
                nticks = 5,
            )

            st.plotly_chart(fig, height=200)

    st.divider()

    ############
    #   Vega   #
    ############

    st.subheader("Vega (bp)")

    write_subtitle(f"As of {value_date.date()}")

    col1, col2 = st.columns(2)

    for n, col in enumerate([col1, col2]):
        model = ["FMM", "HW"][n]

        with col:
            write_subsubtitle(model.replace("HW", "Hull-White"))
            st.dataframe(
                apply_heatmap(
                    strike_cols(
                        vol_pivot["Vega"].loc[value_date].xs(
                            model,
                            level="Model",
                            axis=1
                        ).dropna(
                            how="all",
                            axis=1
                        ).dropna(
                            how="all"
                        ).fillna(0).rename_axis("Tenor") * 10_000
                    ),
                    cmap,
                    axis = None,
                ),
                height="content",
            )

    write_subtitle("Historical evolution")

    vega_tenors = sorted(vol_pivot["Vega"]\
        .index.get_level_values(
            "VolTenor"
        ).unique(), key = lambda x: Tenor(x))

    vega_strikes = sorted(vol_pivot["Vega"]\
        .columns.get_level_values(0).unique())

    col1, col2 = st.columns(2)

    with col1:
        vega_tenor = st.selectbox(
            "Tenor",
            vega_tenors,
            key = "vega_tenor"
        )

    with col2:
        vega_strike = st.selectbox(
            "Strike",
            vega_strikes,
            key = "vega_strike",
            format_func = lambda x: f"{x:.3%}"
        )

    vega_df = vol_pivot["Vega"][vega_strike].xs(
        vega_tenor,
        level="VolTenor"
    )

    fig = go.Figure()

    for n, model in enumerate(["FMM", "HW"]):
        name = model.replace("HW", "Hull-White")
        color = ["dodgerblue", "tomato"][n]

        fig.add_trace(
            go.Scatter(
                x = vega_df.index,
                y = vega_df[model] * 10_000,
                name = name,
                line = dict(
                    color = color,
                    width = 1.1,
                )
            )
        )

    min_y = min(vega_df["FMM"].min(), vega_df["HW"].min()) * 10_000
    max_y = max(vega_df["FMM"].max(), vega_df["HW"].max()) * 10_000

    fig.add_trace(
        go.Scatter(
            x = np.repeat(value_date, 2),
            y = [min_y, max_y],
            name = "Value date",
            mode = "lines",
            line = dict(
                color = "slateblue",
                dash = "dot",
            )
        )
    )

    fig.update_layout(**chart_layout())

    st.plotly_chart(fig, height=200)

    st.divider()

    #############
    #   Volga   #
    #############

    st.subheader("Volga (bp)")

    write_subtitle(f"As of {value_date.date()}")

    col1, col2 = st.columns(2)

    for n, col in enumerate([col1, col2]):
        model = ["FMM", "HW"][n]

        with col:
            write_subsubtitle(model.replace("HW", "Hull-White"))
            st.dataframe(
                apply_heatmap(
                    strike_cols(
                        vol_pivot["Volga"].loc[value_date].xs(
                            model,
                            level="Model",
                            axis=1
                        ).dropna(
                            how="all",
                            axis=1
                        ).dropna(
                            how="all"
                        ).fillna(0).rename_axis("Tenor") * 10_000
                    ),
                    cmap,
                    axis = None,
                ),
                height="content",
            )

    write_subtitle("Historical evolution")

    volga_tenors = sorted(vol_pivot["Volga"]\
        .index.get_level_values(
            "VolTenor"
        ).unique(), key = lambda x: Tenor(x))

    volga_strikes = sorted(vol_pivot["Volga"]\
        .columns.get_level_values(0).unique())

    col1, col2 = st.columns(2)

    with col1:
        volga_tenor = st.selectbox(
            "Tenor",
            volga_tenors,
            key = "volga_tenor"
        )

    with col2:
        volga_strike = st.selectbox(
            "Strike",
            volga_strikes,
            key = "volga_strike",
            format_func = lambda x: f"{x:.3%}"
        )

    volga_df = vol_pivot["Volga"][volga_strike].xs(
        volga_tenor,
        level="VolTenor"
    )

    fig = go.Figure()

    for n, model in enumerate(["FMM", "HW"]):
        name = model.replace("HW", "Hull-White")
        color = ["dodgerblue", "tomato"][n]

        fig.add_trace(
            go.Scatter(
                x = volga_df.index,
                y = volga_df[model] * 10_000,
                name = name,
                line = dict(
                    color = color,
                    width = 1.1,
                )
            )
        )

    min_y = min(volga_df["FMM"].min(), volga_df["HW"].min()) * 10_000
    max_y = max(volga_df["FMM"].max(), volga_df["HW"].max()) * 10_000

    fig.add_trace(
        go.Scatter(
            x = np.repeat(value_date, 2),
            y = [min_y, max_y],
            name = "Value date",
            mode = "lines",
            line = dict(
                color = "slateblue",
                dash = "dot",
            )
        )
    )

    fig.update_layout(**chart_layout())

    st.plotly_chart(fig, height=200)

    st.divider()

    #############
    #   Vanna   #
    #############

    st.subheader("Vanna (bp)")

    write_subtitle(f"As of {value_date.date()}")

    col1, col2 = st.columns(2)

    with col1:
        vanna_curve = st.selectbox(
            "Curve",
            ["Euribor 6M", "ESTR"],
            key = "vanna_curve"
        )

    with col2:
        vanna_rate_tenors = sorted(
            cross_pivot.loc[value_date]\
                [vanna_curve.upper().replace(" ", "")]\
                .dropna(how="all")\
                .index.get_level_values("RateTenor").unique(),
            key = lambda x: Tenor(x)
        )

        vanna_rate_tenor = st.selectbox(
            "Rate tenor",
            vanna_rate_tenors,
            key = "vanna_rate_tenor",
            index = vanna_rate_tenors.index("3Y"),
        )

    col1, col2 = st.columns(2)

    for n, col in enumerate([col1, col2]):
        model = ["FMM", "HW"][n]

        with col:
            write_subsubtitle(model.replace("HW", "Hull-White"))
            st.dataframe(
                apply_heatmap(
                    cross_pivot.loc[value_date][vanna_curve.upper().replace(" ", "")]\
                    .xs(vanna_rate_tenor, level="RateTenor")\
                    .xs(model, level="Model", axis=1).dropna(how = "all").fillna(0) * 10_000,
                    cmap,
                    axis = None,
                ),
                height="content",
            )

    write_subtitle("Historical evolution")

    vanna_rate_tenors = sorted(
        cross_pivot.index.get_level_values("RateTenor").unique(),
        key = lambda x: Tenor(x)
    )

    vanna_vol_tenors = sorted(
        cross_pivot.index.get_level_values("VolTenor").unique(),
        key = lambda x: Tenor(x)
    )

    vanna_strikes = sorted(
        cross_pivot.columns.get_level_values("Strike").unique()
    )
    
    col1, col2, col3 = st.columns(3)

    with col1:
        vanna_rate_tenor_hist = st.selectbox(
            "Rate Tenor",
            vanna_rate_tenors,
            key = "vanna_rate_tenor_hist",
            index = vanna_rate_tenors.index("3Y"),
        )

    with col2:
        vanna_vol_tenor = st.selectbox(
            "Vol Tenor",
            vanna_vol_tenors,
            key = "vanna_vol_tenor",
            index = vanna_vol_tenors.index("3Y"),
        )

    with col3:
        vanna_strike = st.selectbox(
            "Strike",
            vanna_strikes,
            key = "vanna_strike",
            format_func = lambda x: f"{x:.3%}",
        )

    vanna_df = cross_pivot\
        .xs(vanna_strike, level="Strike", axis=1)\
        .xs(vanna_rate_tenor_hist, level="RateTenor")\
        .xs(vanna_vol_tenor, level="VolTenor")

    col1, col2 = st.columns(2)

    for n, col in enumerate([col1, col2]):
        curve = ["Euribor 6M", "ESTR"][n]

        with col:
            write_subsubtitle(curve)
            fig = go.Figure()

            for m, model in enumerate(["FMM", "HW"]):
                color = ["dodgerblue", "tomato"][m]
                fig.add_trace(
                    go.Scatter(
                        x = vanna_df[curve.replace(" ", "").upper()].index,
                        y = vanna_df[curve.replace(" ", "").upper()][model],
                        name = model.replace("HW", "Hull-White"),
                        mode = "lines",
                        line = dict(
                            color = color,
                            width = 1.1,
                        )
                    )
                )

            min_y = min(vanna_df[curve.replace(" ", "").upper()]["FMM"].min(), vanna_df[curve.replace(" ", "").upper()]["HW"].min())
            max_y = max(vanna_df[curve.replace(" ", "").upper()]["FMM"].max(), vanna_df[curve.replace(" ", "").upper()]["HW"].max())

            fig.add_trace(
                go.Scatter(
                    x = np.repeat(value_date, 2),
                    y = [min_y, max_y],
                    name = "Value date",
                    mode = "lines",
                    line = dict(
                        color="slateblue",
                        dash = "dot",
                    )
                )
            )

            fig.update_layout(**chart_layout())

            st.plotly_chart(fig, height=200)

    st.divider()

    st.subheader("Theta")

    write_subtitle(f"As of {value_date.date()}")

    theta_df = risk_table[
        (risk_table.Measure == "Theta")
    ]

    st.dataframe(
        apply_heatmap(
            theta_df[
                theta_df.SensDate == value_date
            ].pivot_table(
                index = "Measure",
                columns = "Model",
                values = "Value",
                aggfunc = "sum"
            ).rename_axis(None)\
            .rename({"HW": "Hull-White"}, axis = 1),
            cmap,
            axis = None,
        ),
        width = 350
    )

    write_subtitle("Historical evolution")

    fig = go.Figure()

    for m, model in enumerate(["FMM", "HW"]):
        color = ["dodgerblue", "tomato"][m]

        fig.add_trace(
            go.Scatter(
                x = theta_df[theta_df.Model == model].SensDate,
                y = theta_df[theta_df.Model == model].Value,
                name = model.replace("HW", "Hull-White"),
                line = dict(
                    color = color,
                    width = 1.1,
                ),
            )
        )

    min_y = min(theta_df[theta_df.Model == "FMM"].Value.min(), theta_df[theta_df.Model == "HW"].Value.min())
    max_y = max(theta_df[theta_df.Model == "FMM"].Value.max(), theta_df[theta_df.Model == "HW"].Value.max())

    fig.add_trace(
        go.Scatter(
            x = np.repeat(value_date, 2),
            y = [min_y, max_y],
            name = "Value date",
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

    st.plotly_chart(fig, height=200)