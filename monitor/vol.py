import streamlit as st
from helpers import date_slider, sort_index_tenor, apply_heatmap
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import streamlit.components.v1 as components

def vol_tab(vols_df, caplet_vol_df, rates_df, cap_validation_df):
    st.subheader("Surfaces")

    vol_surface_date = date_slider(caplet_vol_df.TradeDate.unique(), "vol_surface_dates", label="Valuation Date")
    strikes = [-0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5]
    strikes_div = [i / 100 for i in strikes]
    cap_tenors = ["3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y"]

    col1, col2 = st.columns([0.5, 0.5], gap="medium")

    with col1:
        st.markdown("<h4 style='font-size: 1.1em'>Flat vol surface (Quoted)</h4>", unsafe_allow_html=True)
        df = sort_index_tenor(vols_df.pivot_table(index=["Date", "Tenor"], columns="Strike", values="Vol", aggfunc="sum").loc[vol_surface_date])
        df = df[strikes]
        df.columns = [f"{i / 100:.2%}" for i in df]

        st.dataframe(apply_heatmap(df.loc[cap_tenors], "viridis", "{:.0f}", axis=None, low=0, high=0), height="content", width="stretch")

    with col2:
        st.markdown("<h4 style='font-size: 1.1em'>Stripped vol surface (Caplets)</h4>", unsafe_allow_html=True)
        df = sort_index_tenor(caplet_vol_df[~caplet_vol_df.IsATM].pivot_table(index=["TradeDate", "Tenor"], columns="Strike", values="StrippedVol", aggfunc="sum").loc[vol_surface_date])
        df = df[strikes_div]
        df.columns = [f"{i:.2%}" for i in df.columns]
        df = apply_heatmap(df, "viridis", "{:.0f}", axis=None, low=0, high=0)
        st.dataframe(df, height="content", width="stretch")

    tenor_idx = list(range(len(cap_tenors)))

    colorscale = "Viridis"

    flat_df = vols_df[
        (vols_df.Date == vol_surface_date) &
        (~vols_df.IsATM)
    ].pivot_table(
        index="Tenor",
        columns="Strike",
        values="Vol",
        aggfunc="sum"
    ).loc[cap_tenors][strikes]

    stripped_df = sort_index_tenor(caplet_vol_df[
        (caplet_vol_df.TradeDate == vol_surface_date) &
        (~caplet_vol_df.IsATM) &
        (caplet_vol_df.Strike.isin(strikes_div))
    ].pivot_table(
        index = "Tenor",
        columns = "Strike",
        values = "StrippedVol",
        aggfunc = "sum",
    ))

    stripped_idx = list(range(len(stripped_df.index)))

    fig = make_subplots(
        rows = 1,
        cols = 2,
        specs = [[{"type": "surface"}, {"type": "surface"}]],
        # subplot_titles = ("Flat", "Stripped"),
        horizontal_spacing = 0.05
    )

    fig.add_trace(
        go.Surface(
            x = tenor_idx,
            y = flat_df.columns,
            z = flat_df.values.T,
            colorscale=colorscale,
            showscale=False,
            cmin = flat_df.values.min(),
            cmax = flat_df.values.max(),
            colorbar=dict(
                title=dict(
                    text="bp",
                    font=dict(
                        size=12,
                    ),
                ),
            thickness=10,
            x=1.02,
            ),
        hovertemplate="Strike %{y:.3f}%<br>Tenor: %{x}<br>Vol: %{z}",
        ),
        row = 1,
        col = 1,
    )

    fig.add_trace(
        go.Surface(
            x = stripped_idx,
            y = stripped_df.columns,
            z = stripped_df.values.T,
            colorscale=colorscale,
            cmin = stripped_df.values.min(),
            cmax = stripped_df.values.max(),
            colorbar=dict(
                title=dict(
                    text="bp",
                    font=dict(
                        size=12,
                    ),
                ),
            thickness=10,
            x=1.02,
            ),
        hovertemplate="Strike %{y:.3f}%<br>Tenor: %{x}<br>Vol: %{z}",
        ),
        row = 1,
        col = 2,
    )

    zoom = 0.75

    scene = dict(
        yaxis = dict(
            title = dict(
                text = "Strike (%)",
                font = dict(
                    size = 11,
                ),
            ),
            tickfont = dict(
                size = 10,
            ),
            tickformat = ".2f",
        ),
        zaxis = dict(
            title = dict(
                text = "Vol (bp)",
                font = dict(
                    size = 11
                ),
            ),
            tickfont = dict(
                size = 10,
            ),
            range = [min(flat_df.values.min(), stripped_df.values.min()), max(flat_df.values.max(), stripped_df.values.max())]
        ),
        camera = dict(
            eye = dict(
                x = 1.8 * zoom,
                y = -1.8 * zoom,
                z = 0.6 * zoom,
            ),
            center = dict(
                x = 0,
                y = 0,
                z = -0.3,
            )
        )
    )

    scene1 = dict(
        **scene,
        xaxis = dict(
            title = dict(text = "Tenor", font = dict(
                size = 11
            )),
            tickfont = dict(
                size = 10
            ),
            tickvals = tenor_idx,
            ticktext = cap_tenors,
        )
    )

    scene2 = dict(
        **scene,
        xaxis = dict(
            title = dict(text = "Tenor", font = dict(
                size = 11
            )),
            tickfont = dict(
                size = 10
            ),
            tickvals = stripped_idx,
            ticktext = stripped_df.index,
        )
    )

    fig.update_layout(
        height = 350,
        margin = dict(
            t = 20,
            b = 0,
        ),
        uirevision="const",
        scene = scene1,
        scene2 = scene2,
    )

    # Render as a raw HTML component (not st.plotly_chart) so we can attach a
    # JS listener that mirrors camera rotation/zoom/pan between the two scenes.
    div_id = "vol3d_sync"
    plot_html = fig.to_html(include_plotlyjs="cdn", full_html=False, div_id=div_id)
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
    components.html(font_import + plot_html + sync_script, height=350, scrolling=False)

    st.divider()

    col1, col2 = st.columns(2, gap="medium")

    with col1:

        st.subheader("ATM strikes")

        st.write("Each maturity’s ATM strike is that cap’s forward swap rate, which differs by maturity.")

        atm_df = sort_index_tenor(rates_df[
            (rates_df.Date == vol_surface_date) &
            (rates_df.Curve == "EURIBOR6M")
        ].set_index("Tenor")).loc[cap_tenors].Rate

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x = atm_df.index,
                y = atm_df.values,
                line = dict(
                    shape="spline",
                    smoothing=1,
                    color="slateblue",
                    width=2.5
                ),
                marker = dict(
                    size=8
                ),
                mode = "lines+markers+text",
                text = [f"{i / 100:.2%}" for i in atm_df.values],
                textposition = "bottom center",
                # textfont_size = 15,
                textfont_color = "slateblue",
            )
        )

        fig.update_yaxes(
            title = "Rate (%)",
            tick0 = 0,
            range = [atm_df.values.min() - 0.05, atm_df.values.max() + 0.05],
            nticks = 10,
        )

        fig.update_xaxes(
            title = "Cap Tenor",
        )

        fig.update_layout(
            margin=dict(
                t=0,
                b=0,
            ),
        )

        st.plotly_chart(fig, height=300, width="content")

    with col2:
        st.subheader("Cap repricing validation")
        
        st.write("Repricing a cap with stripped caplet vols and comparing to the flat-vol price.")
    
        atm_validation = abs(cap_validation_df[
            cap_validation_df.IsATM
        ].set_index("Date").RepricingError).resample("W").max()
        
        non_atm_validation = abs(cap_validation_df[~cap_validation_df.IsATM].set_index("Date").RepricingError).resample("W").max()
    
        fig = make_subplots(specs=[[{"secondary_y": True}]])
    
        fig.add_trace(
            go.Scatter(
                x=non_atm_validation.index,
                y=non_atm_validation.values * 1e13,
                name="Non-ATM",
                line=dict(color="dodgerblue", width=1.2)
                ),
            secondary_y=False
        )
    
        fig.add_trace(
            go.Scatter(
                x=atm_validation.index,
                y=atm_validation.values,
                name="ATM",
                line=dict(color="tomato", width=2.2)
                ),
            secondary_y=True,
        )
    
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                xanchor="left",
                x=0,
                y=1,
            ),
            margin=dict(
                t=0,
                b=0,
                l=0,
                r=0,
            )
        )
    
        fig.update_yaxes(
            title_text="ATM repricing error (bp)",
            secondary_y=True,
            tickfont=dict(color="tomato"),
            tick0=0,
            dtick=1,
            showgrid=False,
        )
    
        fig.update_yaxes(
            title_text="Non-ATM repricing error (bp x 1e-13)",
            secondary_y=False,
            tickfont=dict(color="dodgerblue"),
            tick0=0,
            dtick=0.5,
        )
    
        st.plotly_chart(fig, height=300)

    st.divider()

    st.subheader("Surface shape")

    col1, col2 = st.columns([0.5, 0.5], gap="medium")

    with col1:
        st.markdown("<h4 style='font-size: 1.1em'>Term structure</h4>", unsafe_allow_html=True)

        strike = st.selectbox("Strike", strikes_div, strikes_div.index(0.02), lambda x: f"{x:.3%}")

        df_flat = sort_index_tenor(
            vols_df[
                (vols_df.Date == vol_surface_date) &
                (vols_df.Strike == strike * 100)
            ].set_index("Tenor").loc[cap_tenors].Vol
        )

        df_stripped = sort_index_tenor(
            caplet_vol_df[
                ~caplet_vol_df.IsATM
            ].pivot_table(
                index=["TradeDate", "Strike", "Tenor"],
                values="StrippedVol"
            ).loc[(vol_surface_date, strike)].loc[cap_tenors]
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=df_flat.index,
                y=df_flat.values,
                mode="lines+markers",
                name="Flat",
                line=dict(shape="spline", smoothing=1, color="dodgerblue", width=2.5),
                marker=dict(size=8)
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df_stripped.index,
                y=df_stripped.StrippedVol,
                mode="lines+markers",
                name="Stripped",
                line=dict(shape="spline", smoothing=1, color="tomato", width=2.5),
                marker=dict(size=8)
            )
        )

        fig.update_layout(
            yaxis_title="Vol (bp)",
            margin=dict(
                t=0,
                b=0,
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
            nticks=10,
        )

        fig.update_xaxes(
            title="Maturity",
        )

        st.plotly_chart(fig, width="stretch", height=300)
    
    with col2:
        st.markdown("<h4 style='font-size: 1.1em'>Smile</h4>", unsafe_allow_html=True)

        smile_tenor = st.selectbox("Tenor", cap_tenors, index=1)

        df_flat = vols_df[
            (vols_df.Date == vol_surface_date) &
            (vols_df.Tenor == smile_tenor) &
            (~vols_df.IsATM)
        ].set_index("Strike").loc[strikes].Vol.copy()

        df_stripped = caplet_vol_df[
            (caplet_vol_df.TradeDate == vol_surface_date) & 
            (caplet_vol_df.Tenor == smile_tenor)
        ].set_index("Strike").loc[strikes_div]

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=df_flat.index,
                y=df_flat.values,
                mode="lines+markers",
                name="Flat",
                line=dict(shape="spline", smoothing=1, color="dodgerblue", width=2.5),
                marker=dict(size=8, symbol="circle", color="dodgerblue")
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df_stripped.index * 100,
                y=df_stripped.StrippedVol,
                mode="lines+markers",
                name="Stripped",
                line=dict(shape="spline", smoothing=1, color="tomato", width=2.5),
                marker=dict(size=8, symbol="circle", color="tomato")
            )
        )

        fig.update_layout(
            yaxis_title="Vol (bp)",
            margin=dict(
                t=0,
                b=0,
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                xanchor="left",
                x=0,
                y=1,
            ),
        )

        fig.update_xaxes(
            tick0=-0.5,
            dtick = 0.5,
            title="Strike (%)"
        )

        fig.update_yaxes(
            nticks=10,
        )
        

        st.plotly_chart(fig, width="stretch", height=300)

    st.divider()