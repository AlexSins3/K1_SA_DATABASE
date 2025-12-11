# app.py

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from tabs.dataset_view import show_dataset_tab
from tabs.athlete_focus import show_athlete_focus_tab
from tabs.graphs import show_graphs_tab
from tabs.acm import show_acm_tab


# =========================
# Configuration Streamlit
# =========================
st.set_page_config(
    page_title="Analyse Katas K1 / SA",
    layout="wide",
)

DATA_PATH = Path("data") / "Database_K1_SA.csv"


# =========================
# Chargement & préparation des données
# =========================
@st.cache_data
def load_data(csv_path: Path) -> pd.DataFrame:
    """
    Charge la base de données des katas et prépare les types.
    La même fonction fonctionnera pour les futures BDD
    tant que la structure des colonnes reste identique.
    """
    df = pd.read_csv(csv_path, sep=';', encoding='utf-8')

    # Normalisation de la colonne Note (virgule -> point si besoin)
    if df['Note'].dtype == object:
        df['Note'] = (
            df['Note']
            .astype(str)
            .str.replace(',', '.', regex=False)
        )

    # Conversion des variables numériques principales
    for col in ['Age', 'Ranking', 'Note', 'Year']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Conversion de Victoire en booléen si ce n'est pas déjà le cas
    if 'Victoire' in df.columns and df['Victoire'].dtype != bool:
        df['Victoire'] = df['Victoire'].astype(str).str.lower().isin(['true', '1', 'vrai', 'yes'])

    # Mise en catégories pour toutes les colonnes non numériques
    numeric_cols = [
        col for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].dtype != bool
    ]
    categorical_cols = [col for col in df.columns if col not in numeric_cols]

    for col in categorical_cols:
        df[col] = df[col].astype('string').astype('category')

    return df


def add_footer():
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


def main():
    st.title("Suivi & Analyse des Katas – Premier League (K1) & Series A (SA)")

    # Chargement des données
    if not DATA_PATH.exists():
        st.error(f"Fichier de données introuvable : {DATA_PATH}")
        st.stop()

    data = load_data(DATA_PATH)

    # Onglets principaux
    tab1, tab2, tab3, tab4 = st.tabs([
        "Dataset",
        "Focus Athlète",
        "Générateur de graphiques interactifs",
        "Analyse des Correspondances Multiples (ACM)"
    ])

    with tab1:
        show_dataset_tab(data)

    with tab2:
        show_athlete_focus_tab(data)

    with tab3:
        show_graphs_tab(data)

    with tab4:
        show_acm_tab(data)

    # Footer global
    add_footer()


if __name__ == "__main__":
    main()
