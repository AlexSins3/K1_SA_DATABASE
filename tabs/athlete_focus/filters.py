# tabs/athlete_focus/filters.py

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import streamlit as st

from utils.ui import filter_panel_open, filter_panel_close
from utils.lang import t


@dataclass
class AthleteFilterState:
    """All state produced by the filter panel and consumed by charts / history."""
    full_data: pd.DataFrame
    selected_type_compet: str
    selected_sexe: str
    selected_athlete: str
    selected_compare_athlete: str
    athlete_data: pd.DataFrame
    compare_athlete_data: pd.DataFrame | None
    selected_tours: list = field(default_factory=list)
    selected_competitions: list = field(default_factory=list)


def render_filters(data: pd.DataFrame) -> AthleteFilterState | None:
    """Render the filter sidebar and return an ``AthleteFilterState`` (or ``None`` on error)."""

    df = data.copy()
    filter_panel_open()
    st.markdown(t("### 🎯 Filtres"))

    # ── Type de compétition ──
    type_compet_options = [t("Tous"), "Premier League (K1)", "Series A (SA)"]
    selected_type_compet = st.radio(
        t("Type de compétition"),
        type_compet_options,
        key="athlete_type_compet",
    )

    if selected_type_compet == "Premier League (K1)":
        data_athlete = df[df["Type_Compet"] == "K1"].copy()
    elif selected_type_compet == "Series A (SA)":
        data_athlete = df[df["Type_Compet"] == "SA"].copy()
    else:
        data_athlete = df.copy()

    # ── Sexe ──
    sexe_options = sorted(data_athlete["Sexe"].dropna().unique().tolist())
    if not sexe_options:
        st.warning(t("Aucun sexe disponible dans les données filtrées."))
        filter_panel_close()
        return None

    selected_sexe = st.radio(t("Sexe des athlètes"), sexe_options, key="athlete_sexe")
    data_athlete = data_athlete[data_athlete["Sexe"] == selected_sexe].copy()

    # ── Athlète principal ──
    athlete_names = sorted(data_athlete["Nom"].dropna().unique().tolist())
    if not athlete_names:
        st.warning(t("Aucun athlète disponible avec les filtres sélectionnés."))
        filter_panel_close()
        return None

    selected_athlete = st.selectbox(t("Athlète principal"), athlete_names, index=None, placeholder=t("Choisir un athlète..."), key="athlete_main_select")
    if selected_athlete is None:
        st.info(t("Sélectionnez un athlète pour afficher son profil."))
        filter_panel_close()
        return None
    athlete_data = data_athlete[data_athlete["Nom"] == selected_athlete].copy()

    # ── Athlète comparé ──
    compare_options = [t("Aucun")] + [n for n in athlete_names if n != selected_athlete]
    selected_compare_athlete = st.selectbox(t("Comparer à"), compare_options, key="athlete_compare_select")

    compare_athlete_data = None
    if selected_compare_athlete != t("Aucun"):
        compare_athlete_data = data_athlete[data_athlete["Nom"] == selected_compare_athlete].copy()

    st.markdown("---")
    st.markdown(t("#### 🔎 Filtres avancés"))

    # ── Tours ──
    if compare_athlete_data is not None:
        tour_options = sorted(
            set(athlete_data["N_Tour"].dropna().unique())
            | set(compare_athlete_data["N_Tour"].dropna().unique())
        )
    else:
        tour_options = sorted(athlete_data["N_Tour"].dropna().unique().tolist())

    selected_tours = []
    if tour_options:
        selected_tours = st.multiselect(
            t("Tours (N_Tour)"), options=tour_options, default=tour_options, key="athlete_tours_filter",
        )

    # ── Compétitions ──
    if compare_athlete_data is not None:
        competition_options = sorted(
            set(athlete_data["Competition"].dropna().unique())
            | set(compare_athlete_data["Competition"].dropna().unique())
        )
    else:
        competition_options = sorted(athlete_data["Competition"].dropna().unique().tolist())

    selected_competitions = []
    if competition_options:
        selected_competitions = st.multiselect(
            t("Compétitions"), options=competition_options, default=competition_options, key="athlete_compet_filter",
        )

    filter_panel_close()

    return AthleteFilterState(
        full_data=df,
        selected_type_compet=selected_type_compet,
        selected_sexe=selected_sexe,
        selected_athlete=selected_athlete,
        selected_compare_athlete=selected_compare_athlete,
        athlete_data=athlete_data,
        compare_athlete_data=compare_athlete_data,
        selected_tours=selected_tours,
        selected_competitions=selected_competitions,
    )
