# tabs/dataset_view.py

import pandas as pd
import streamlit as st


def _convertir_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode('utf-8')


def show_dataset_tab(data: pd.DataFrame) -> None:
    st.header("Affichage du Dataset de Karaté")

    # On travaille sur une copie pour ne jamais modifier l'original
    df = data.copy()

    colonnes = df.columns.tolist()
    colonnes_selectionnees = st.multiselect(
        "Sélectionnez les colonnes à afficher",
        colonnes,
        default=colonnes,
    )

    # Exemple de filtre conditionnel (si Grade existe dans la BDD)
    if 'Grade' in df.columns:
        grades = df['Grade'].dropna().unique().tolist()
        grade_selection = st.multiselect("Filtrer par Grade", grades)
        if grade_selection:
            df = df[df['Grade'].isin(grade_selection)]

    st.dataframe(df[colonnes_selectionnees], width="stretch")

    # Bouton de téléchargement du CSV filtré
    csv = _convertir_csv(df[colonnes_selectionnees])
    st.download_button(
        label="Télécharger le CSV filtré",
        data=csv,
        file_name='dataset_karate_filtre.csv',
        mime='text/csv',
    )
