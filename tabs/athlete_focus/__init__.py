# tabs/athlete_focus/__init__.py

import streamlit as st

from tabs.athlete_focus.filters import render_filters
from tabs.athlete_focus.charts import (
    render_athlete_info,
    render_max_tour,
    render_kata_histogram,
    render_kiviat_tour,
    render_kiviat_kata,
    render_flag_radar_tour,
    render_flag_radar_kata,
)
from tabs.athlete_focus.history import render_history
from utils.interpretations import show_tab_help
from utils.lang import t


@st.fragment
def show_athlete_focus_tab(data):
    st.header(t("Focus Athlète"))
    show_tab_help("athlete_focus")

    filters_col, content_col = st.columns([0.9, 2.4])

    with filters_col:
        state = render_filters(data)

    if state is None:
        return

    with content_col:
        if state.athlete_data.empty:
            st.warning(t("Aucune donnée pour l'athlète sélectionné."))
            return

        render_athlete_info(state)
        render_max_tour(state)
        render_kata_histogram(state)
        render_kiviat_tour(state)
        render_flag_radar_tour(state)
        render_kiviat_kata(state)
        render_flag_radar_kata(state)
        render_history(state)
