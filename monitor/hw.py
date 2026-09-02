import streamlit as st
import numpy as np
from helpers import date_slider, chart_layout
import plotly.graph_objects as go

def hull_white_tab(hw_df):
    st.subheader("Calibration results")
    hw_date = date_slider(
        sorted(hw_df.TradeDate.unique()),
        key = "hw_date",
        label = "Trade date"
    )

    try:
        prev_date = hw_df[hw_df.TradeDate < hw_date].iloc[-1].TradeDate
    except:
        prev_date = hw_date

    hw_df = hw_df.set_index("TradeDate")

    curr_a = hw_df.loc[hw_date].a
    prev_a = hw_df.loc[prev_date].a
    a_diff = curr_a - prev_a

    curr_sigma = hw_df.loc[hw_date].sigma
    prev_sigma = hw_df.loc[prev_date].sigma
    sigma_diff = curr_sigma - prev_sigma

    curr_err = hw_df.loc[hw_date].ResidualError
    prev_err = hw_df.loc[prev_date].ResidualError
    err_diff = curr_err - prev_err

    at_bound = hw_df.loc[hw_date].AtBound

    st.markdown(
        """
        <style>
        /* Change the value font size */
        [data-testid="stMetricValue"] {
            font-size: 2em;
        }
        /* Change the label font size */
        [data-testid="stMetricLabel"] {
            font-size: 1.2em;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("$a$: mean reversion", f"{curr_a:.4f}", f"{a_diff * 10_000:.2f} bp")

    col2.metric(r"$\sigma_\text{HW}$: instantaneous vol", f"{curr_sigma * 10_000:.2f} bp", f"{sigma_diff * 10_000:.2f} bp")

    col3.metric("SSE: residual error", f"{curr_err * 10_000:.4f} bp", f"{err_diff * 10_000:.4f} bp")

    col4.metric("Interior convergence", "⚠️ At bound" if at_bound else "✅ Interior")

    if not at_bound:
        st.write(f"Mean-reversion half-life: {np.log(2) / curr_a:.1f} years.")

    st.divider()

    st.subheader("Historical parameters")

    col1, col2 = st.columns(2)

    for n, col in enumerate([col1, col2]):
        title = ["$a$: mean reversion", r"$\sigma$: instantaneous vol"][n]
        measure = ["a", "sigma"][n]

        with col:
            st.write(title)

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x = hw_df.index,
                    y = hw_df[measure],
                    name = measure.replace("sigma", "σ"),
                    mode = "lines",
                    line = dict(
                        color = "dodgerblue",
                        width = 0.5,
                    )
                )
            )

            fig.add_trace(
                go.Scatter(
                    x = hw_df[hw_df.AtBound].index,
                    y = hw_df[hw_df.AtBound][measure],
                    name = "At bound",
                    mode = "markers",
                    marker = dict(
                        color = "tomato",
                        size = 2.5,
                    )
                )
            )

            fig.add_trace(
                go.Scatter(
                    x = hw_df[~hw_df.AtBound].index,
                    y = hw_df[~hw_df.AtBound][measure],
                    name = "Interior convergence",
                    mode = "markers",
                    marker = dict(
                        color = "dodgerblue",
                        size = 2.5,
                    )
                )
            )

            min_y = min(hw_df[hw_df.AtBound][measure].min(), hw_df[~hw_df.AtBound][measure].min())
            max_y = max(hw_df[hw_df.AtBound][measure].max(), hw_df[~hw_df.AtBound][measure].max())

            fig.add_trace(
                go.Scatter(
                    x = np.repeat(hw_date, 2),
                    y = [min_y, max_y],
                    name = "Trade date",
                    mode = "lines",
                    line = dict(
                        color = "slateblue",
                        dash = "dot",
                    )
                )
            )

            fig.update_yaxes(
                nticks = 10,
            )

            fig.update_layout(
                **chart_layout()
            )

            st.plotly_chart(fig, key=title, height=300)

    st.write("Red points = calibration pinned at a bound (mostly the pre-2022 negative-rate window). Dotted line marks the selected date.")

    st.divider()

    st.subheader("Residual calibration error (SSE)")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x = hw_df.index,
            y = hw_df.ResidualError,
            name = "SSE",
            mode = "lines",
            line = dict(
                color = "dodgerblue",
                width = 0.5
            )
        ),
    )

    fig.add_trace(
        go.Scatter(
            x = hw_df[~hw_df.AtBound].index,
            y = hw_df[~hw_df.AtBound].ResidualError,
            name = "Interior convergence",
            mode = "markers",
            marker = dict(
                color = "dodgerblue",
                size = 2.5,
            ),
        ),
    )

    fig.add_trace(
        go.Scatter(
            x = hw_df[hw_df.AtBound].index,
            y = hw_df[hw_df.AtBound].ResidualError,
            name = "At bound",
            mode = "markers",
            marker = dict(
                color = "tomato",
                size = 2.5,
            )
        ),
    )

    fig.update_layout(
        **chart_layout(),
        yaxis = dict(
            type = "log",
            title = "SSE (log scale)"
        )
    )

    st.plotly_chart(fig, height = 300)