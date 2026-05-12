# tabs/dataset_view.py

import pandas as pd
import streamlit as st

from utils.ui import filter_panel_open, filter_panel_close
from utils.interpretations import show_tab_help
from utils.display import fmt_col, format_display_df


def _convertir_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


@st.fragment
def show_dataset_tab(data: pd.DataFrame) -> None:
    st.header("Affichage du Dataset de Karaté")
    show_tab_help("dataset")

    df = data.copy()

    # ── Layout : filtres à gauche, contenu à droite ──
    filters_col, content_col = st.columns([0.9, 2.4])

    with filters_col:
        filter_panel_open()
        st.markdown("### 🎛️ Filtres")

        # ---- Type de compétition ----
        type_compet_options = ["Tous", "Premier League (K1)", "Series A (SA)"]
        selected_type = st.radio(
            "Type de compétition",
            type_compet_options,
            key="dataset_type_compet",
        )
        if selected_type == "Premier League (K1)":
            df = df[df["Type_Compet"] == "K1"]
        elif selected_type == "Series A (SA)":
            df = df[df["Type_Compet"] == "SA"]

        # ---- Année ----
        if "Year" in df.columns:
            years = sorted(df["Year"].dropna().unique().tolist())
            if years:
                selected_years = st.multiselect(
                    "Année(s)",
                    options=years,
                    default=years,
                    key="dataset_year_filter",
                )
                if selected_years:
                    df = df[df["Year"].isin(selected_years)]

        # ---- Sexe ----
        if "Sexe" in df.columns:
            sexes = sorted(df["Sexe"].dropna().unique().tolist())
            if sexes:
                selected_sexes = st.multiselect(
                    "Sexe",
                    options=sexes,
                    default=sexes,
                    key="dataset_sexe_filter",
                )
                if selected_sexes:
                    df = df[df["Sexe"].isin(selected_sexes)]

        # ---- Grade ----
        if "Grade" in df.columns:
            grades = sorted(df["Grade"].dropna().unique().tolist())
            if grades:
                grade_selection = st.multiselect(
                    "Grade",
                    options=grades,
                    key="dataset_grade_filter",
                )
                if grade_selection:
                    df = df[df["Grade"].isin(grade_selection)]

        # ---- Competition ----
        if "Competition" in df.columns:
            competitions = sorted(df["Competition"].dropna().unique().tolist())
            if competitions:
                selected_compets = st.multiselect(
                    "Compétition(s)",
                    options=competitions,
                    key="dataset_compet_filter",
                )
                if selected_compets:
                    df = df[df["Competition"].isin(selected_compets)]

        # ---- Recherche athlète ----
        st.markdown("---")
        search_name = st.text_input("Rechercher un athlète", key="dataset_search_name")
        if search_name:
            df = df[df["Nom"].astype(str).str.contains(search_name, case=False, na=False)]

        filter_panel_close()

    with content_col:
        # ── Statistiques résumées ──
        st.subheader("Résumé")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Lignes", len(df), help="Nombre total de lignes dans le dataset filtré")
        c2.metric("Athlètes", df["Nom"].nunique() if "Nom" in df.columns else "–", help="Nombre d'athlètes distincts dans la sélection")
        c3.metric("Compétitions", df["Competition"].nunique() if "Competition" in df.columns else "–", help="Nombre de compétitions différentes")
        c4.metric("Katas distincts", df["Kata"].nunique() if "Kata" in df.columns else "–", help="Nombre de katas différents joués")

        # ── Sélection de colonnes ──
        colonnes = df.columns.tolist()
        colonnes_selectionnees = st.multiselect(
            "Colonnes à afficher",
            options=colonnes,
            default=colonnes,
            format_func=fmt_col,
            key="dataset_cols_selection",
        )
        if not colonnes_selectionnees:
            colonnes_selectionnees = colonnes

        # ── Affichage ──
        st.dataframe(format_display_df(df[colonnes_selectionnees]), use_container_width=True)

        # ── Téléchargement ──
        csv = _convertir_csv(df[colonnes_selectionnees])
        st.download_button(
            label="Télécharger le CSV filtré",
            data=csv,
            file_name="dataset_karate_filtre.csv",
            mime="text/csv",
            key="dataset_download_button",
        )
