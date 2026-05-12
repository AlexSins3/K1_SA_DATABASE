"""Shared Streamlit UI components (CSS, filter panels, footer)."""

import html as html_lib

import streamlit as st

FILTER_PANEL_CSS = """
<style>
.filter-panel {
    background-color: #f6f8ff;
    padding: 1rem 1.2rem;
    border-radius: 0.8rem;
    border: 1px solid #d9e1ff;
}
.filter-panel h3, .filter-panel h4 {
    margin-top: 0.2rem;
    margin-bottom: 0.6rem;
}
.filter-panel label {
    font-weight: 500;
}
</style>
"""


def filter_panel_open():
    """Inject shared CSS and open the filter panel ``<div>``."""
    st.markdown(FILTER_PANEL_CSS, unsafe_allow_html=True)
    st.markdown("<div class='filter-panel'>", unsafe_allow_html=True)


def filter_panel_close():
    """Close the filter panel ``<div>``."""
    st.markdown("</div>", unsafe_allow_html=True)


def type_compet_radio(data, key_prefix: str):
    """Render a Type-compétition radio and return (label, filtered_df)."""
    options = ["Tous", "Premier League (K1)", "Series A (SA)"]
    selected = st.radio("Type de compétition", options, key=f"{key_prefix}_type_compet")

    if selected == "Premier League (K1)":
        filtered = data[data["Type_Compet"] == "K1"].copy()
    elif selected == "Series A (SA)":
        filtered = data[data["Type_Compet"] == "SA"].copy()
    else:
        filtered = data.copy()

    return selected, filtered


def athlete_label_html(name: str) -> str:
    """Return a safe HTML snippet displaying an athlete's name."""
    safe_name = html_lib.escape(str(name))
    return f"<p style='font-size:13px;font-style:italic;color:#555;'>Athlète : {safe_name}</p>"


def highlight_victory_series(s):
    """Styler function: green for 'Oui', red for 'Non'."""
    styles = []
    for v in s:
        if v == "Oui":
            styles.append("background-color: #d4edda;")
        elif v == "Non":
            styles.append("background-color: #f8d7da;")
        else:
            styles.append("")
    return styles


def add_footer():
    """Render the global application footer."""
    footer = """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        right: 0;
        text-align: right;
        width: 100%;
        padding: 10px;
        font-size: 12px;
        color: grey;
        z-index: 9999;
    }
    .footer .source {
        color: lightgrey;
        opacity: 0.7;
    }
    .footer p {
        margin: 0;
    }
    </style>
    <div class="footer">
        <p class="source">Source : SportData</p>
        <p>&copy; Alexis Vincent</p>
    </div>
    """
    st.markdown(footer, unsafe_allow_html=True)
