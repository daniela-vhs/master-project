import streamlit as st
from helpers import date_slider, write_subtitle, chart_layout
from quant.dates import Tenor
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dateutil.relativedelta import relativedelta as tdelta
import numpy as np

def curve_tab(rates_df, zero_rates_df, curve_validation_df):
    rate_date = date_slider(
        sorted(zero_rates_df.TradeDate.unique()),
        key = "rate_date"   
    )

    par_pivot = rates_df.pivot(
        index = ["Date", "Tenor"],
        columns = "Curve",
        values = "Rate"
    ) / 100

    zero_pivot = zero_rates_df.pivot(
        index = ["TradeDate", "Tenor"],
        columns = "Curve",
        values = "ZeroRate"
    )

    st.subheader("Par and zero-rate curves")

    col1, col2 = st.columns(2)

    for n, col in enumerate([col1, col2]):
        curve = ["Euribor 6M", "ESTR"][n]
        with col:
            write_subtitle(curve)

            common_tenors = sorted(set(par_pivot.loc[rate_date][curve.replace(" ", "").upper()].dropna().index) & set(zero_pivot.loc[rate_date][curve.replace(" ", "").upper()].dropna().index), key = lambda x: Tenor(x))

            zero_rates = zero_pivot.loc[rate_date][curve.replace(" ", "").upper()].loc[common_tenors]

            par_rates = par_pivot.loc[rate_date][curve.replace(" ", "").upper()].loc[common_tenors]

            maturities = [rate_date + tdelta(days = int(Tenor(i).months * 30)) for i in common_tenors]

            fig = go.Figure()

            for n, rate_curve in enumerate([par_rates, zero_rates]):
                name = ["Par", "Zero"][n]
                color = ["dodgerblue", "tomato"][n]

                fig.add_trace(
                    go.Scatter(
                        x = maturities,
                        y = np.round(rate_curve * 100, 4),
                        name = name,
                        mode = "lines+markers",
                        line = dict(
                            color = color,
                            shape = "spline",
                            smoothing = 1,
                        ),
                    )
                )

            fig.update_xaxes(
                labelalias = common_tenors,
                title = "Maturity",
            )

            fig.update_yaxes(
                title = "Rate (%)",
            )

            fig.update_layout(
                **chart_layout()
            )

            st.plotly_chart(fig, height=300)

    st.divider()

    st.subheader("Historical zero rate: single tenor")

    common_tenors = sorted(
        set(zero_pivot.ESTR.dropna().index.get_level_values("Tenor")) & set(zero_pivot.EURIBOR6M.dropna().index.get_level_values("Tenor")),
        key = lambda x: Tenor(x)
    )[2:]

    zero_tenor = st.selectbox(
        "Tenor",
        common_tenors,
        key = "zero_tenor"
    )

    zero_tenor_df = zero_pivot.xs(
        zero_tenor,
        level = "Tenor"
    ).sort_index()

    fig = go.Figure()

    for n, curve in enumerate(["ESTR", "Euribor 6M"]):
        color = ["dodgerblue", "tomato"][n]
        fig.add_trace(
            go.Scatter(
                x = zero_tenor_df.index,
                y = zero_tenor_df[curve.replace(" ", "").upper()],
                name = curve,
                mode = "lines",
                line = dict(
                    color = color,
                    width = 1.1,
                )
            )
        )

    min_y = min(zero_tenor_df["ESTR"].min(), zero_tenor_df["EURIBOR6M"].min())
    max_y = max(zero_tenor_df["ESTR"].max(), zero_tenor_df["EURIBOR6M"].max())

    fig.add_trace(
        go.Scatter(
            x = np.repeat(rate_date, 2),
            y = [min_y, max_y],
            mode = "lines",
            line = dict(
                color = "slateblue",
                dash = "dot",
            ),
            name = "Value date"
        )
    )

    fig.update_layout(
        **chart_layout()
    )

    st.plotly_chart(fig, height=250)

    st.divider()

    st.subheader("Bootstrap validation – IRS repricing error")

    st.markdown("Feeding each curve's own bootstrapped instruments back through the pricer.<br>Error = recovered par rate − quoted rate, in bp.", unsafe_allow_html=True)

    validation_pivot = curve_validation_df.copy()
    validation_pivot[["EstrError", "EurError"]] = abs(validation_pivot[["EstrError", "EurError"]])
    validation_pivot = validation_pivot.pivot_table(
        index = "TradeDate",
        values = ["EstrError", "EurError"],
        aggfunc = "max"
    )

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x = validation_pivot.index,
            y = validation_pivot.EstrError * 1e4,
            name = "ESTR repricing error",
            line = dict(
                color = "dodgerblue",
                width = 1.1,
            )
        ),
        secondary_y = False,
    )

    fig.add_trace(
        go.Scatter(
            x = validation_pivot.index,
            y = validation_pivot.EurError * 1e2,
            name = "Euribor 6M repricing error",
            line = dict(
                color = "tomato",
                width = 1.1,
            )
        ),
        secondary_y = True,
    )

    min_y = (validation_pivot.EstrError * 1e4).min()
    max_y = (validation_pivot.EstrError * 1e4).max()

    fig.add_trace(
        go.Scatter(
            x = np.repeat(rate_date, 2),
            y = [min_y, max_y],
            mode = "lines",
            line = dict(
                color = "slateblue",
                dash = "dot",
            ),
            name = "Value date"
        )
    )

    fig.update_yaxes(
        secondary_y=True,
        title = dict(
            text = r"Repricing error (bp x 10<sup>2</sup>)",
        ),
        tickfont = dict(
            color = "tomato",
        ),
        nticks = 10,
        showgrid = False,
    )

    fig.update_yaxes(
        secondary_y=False,
        title = dict(
            text = r"Repricing error (bp x 10<sup>4</sup>)",
        ),
        tickfont = dict(
            color = "dodgerblue",
        ),
        nticks = 10,
    )

    fig.update_layout(
        **chart_layout(),
    )

    st.plotly_chart(fig, height=300)