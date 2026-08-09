import streamlit as st
import json
from helpers import write_subtitle, write_subsubtitle

@st.cache_data
def load_estr_convention():
    with open("../market_conventions/estr.json") as f:
        return json.loads(f.read())

@st.cache_data
def load_euribor_convention():
    with open("../market_conventions/euribor6m.json") as f:
        return json.loads(f.read())

@st.cache_data
def load_vol_convention():
    with open("../market_conventions/vol_surface.json") as f:
        return json.loads(f.read())


def conventions_tab():
    st.subheader("Market conventions & data source")

    estr_convention    = load_estr_convention()
    euribor_convention = load_euribor_convention()

    col1, col2 = st.columns(2, gap="large")

    st.markdown(
        f"""
        <style>
        .container {{
            border-radius: 4pt;
            padding: 0.5em 0 0.5em 0;
            font-size: 0.9em;
            border: 1px solid rgba(205, 209, 212, 0.5);
            background-color: rgba(205, 209, 212, 0.1);
            margin-bottom: 1em;
            display: table;
            width: 100%;
        }}
        .row {{
            display: table-row;
        }}
        .row div {{
            padding-top: 0.2em;
            padding-bottom: 0.2em;
        }}
        .left {{
            padding-left: 1em;
            color: slategrey;
            font-weight: bold;
            display: table-cell;
        }}
        .value {{
            padding-right: 1em;
            text-align: right;
            overflow: auto;
            min-height: 20;
            display: table-cell;
            padding-left: 1em;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    for i, col in enumerate([col1, col2]):
        curve = ["Euribor 6M", "ESTR"][i]
        conv  = [euribor_convention, estr_convention][i]

        with col:
            write_subtitle(curve)
            write_subsubtitle("Overview")

            render = '<div class="container">'

            for key in ["curve", "description", "settlement", "discounting"]:
                render += f'<div class="row"><div class="left">{key.title()}</div><div class="value">{conv[key]}</div></div>'

            render += "</div>"

            st.markdown(render, unsafe_allow_html=True)
            
            write_subsubtitle("Fixed leg")

            fixed_leg_data = conv["fixed_leg"]

            render = '<div class="container">'

            for key in fixed_leg_data.keys():
                if key == "currency":
                    continue

                render += f'<div class="row"><div class="left">{key.replace("_", " ").title()}</div><div class="value">{fixed_leg_data[key]}</div></div>'

            render += "</div>"

            st.markdown(render, unsafe_allow_html=True)

            write_subsubtitle("Float leg")

            fixed_leg_data = conv["float_leg"]
            
            render = '<div class="container">'

            for key in fixed_leg_data.keys():
                if key == "currency":
                    continue

                render += f'<div class="row"><div class="left">{key.replace("_", " ").title()}</div><div class="value">{fixed_leg_data[key]}</div></div>'

            render += "</div>"

            st.markdown(render, unsafe_allow_html=True)

            # write_subsubtitle("Tickers (Bloomberg)")

            # instruments = conv["instruments"]

            # render = '<div class="container">'

            # for instrument in instruments:
            #     render += f'<div class="row"><div class="left">{instrument["tenor"]}</div><div class="value">{instrument["ticker"]}</div></div>'

            # render += "</div>"

            # st.markdown(render, unsafe_allow_html=True)

    st.divider()

    st.subheader("Cap vol surface")

    vol_convention = load_vol_convention()

    col1, col2 = st.columns(2, gap="large")

    with col1:
        write_subsubtitle("Overview")
        render = """<div class="container">"""

        for key in ["surface", "description", "vol_unit", "settlement", "option_style", "discounting"]:
            render += f"""<div class="row"><div class="left">{key.replace("_", " ").title()}</div><div class="value">{vol_convention[key].split(".")[0]}</div></div>"""

        render += "</div>"

        st.markdown(render, unsafe_allow_html=True)

    with col2:
        write_subsubtitle("Underlying index")
        render = """<div class="container">"""
        
        for key in vol_convention["underlying_indices"]["EUR006M"].keys():
            render += f"""<div class="row"><div class="left">{key.replace("_", " ").title()}</div><div class="value">{vol_convention["underlying_indices"]["EUR006M"][key].split(".")[0]}</div></div>"""

        render += "</div>"

        st.markdown(render, unsafe_allow_html=True)