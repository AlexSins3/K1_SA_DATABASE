# tabs/dataset_view.py

import pandas as pd
import streamlit as st


def _convertir_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def show_dataset_tab(data: pd.DataFrame) -> None:
    st.header("Affichage du Dataset de Karaté")

    # On travaille sur une copie pour ne jamais modifier l'original
    df = data.copy()

    # === Sélection des colonnes ===
    colonnes = df.columns.tolist()
    colonnes_selectionnees = st.multiselect(
        "Sélectionnez les colonnes à afficher",
        options=colonnes,
        default=colonnes,
        key="dataset_cols_selection",  # 👈 clé unique pour ce widget
    )

    # Si l'utilisateur décoche tout, on évite de crasher → on affiche tout
    if not colonnes_selectionnees:
        colonnes_selectionnees = colonnes

    # === Filtre optionnel sur Grade ===
    if "Grade" in df.columns:
        grades = (
            df["Grade"]
            .dropna()
            .unique()
            .tolist()
        )
        grades.sort()

        grade_selection = st.multiselect(
            "Filtrer par Grade",
            options=grades,
            key="dataset_grade_filter",  # 👈 clé unique aussi
        )

        if grade_selection:
            df = df[df["Grade"].isin(grade_selection)]

    # === Affichage du dataframe ===
    st.dataframe(df[colonnes_selectionnees], width="stretch")

    # === Bouton de téléchargement du CSV filtré ===
    csv = _convertir_csv(df[colonnes_selectionnees])
    st.download_button(
        label="Télécharger le CSV filtré",
        data=csv,
        file_name="dataset_karate_filtre.csv",
        mime="text/csv",
        key="dataset_download_button",  # 👈 clé pour être 100% safe
    )
