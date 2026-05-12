# app.py

from pathlib import Path

import pandas as pd
import streamlit as st

from config import DATA_PATH
from utils.ui import add_footer
from utils.interpretations import show_glossaire
from utils.display import fmt_athlete, fmt_underscore
from tabs.dataset_view import show_dataset_tab
from tabs.athlete_focus import show_athlete_focus_tab
from tabs.graphs import show_graphs_tab
from tabs.acm import show_acm_tab
from tabs.proba_victoire_kata import show_proba_victoire_kata_tab
from tabs.progression import show_progression_tab
from tabs.score_differential import show_score_differential_tab
from tabs.continental import show_continental_tab
from tabs.tour_advancement import show_tour_advancement_tab
from tabs.kata_diversity import show_kata_diversity_tab


# =========================
# Configuration Streamlit
# =========================
st.set_page_config(
    page_title="Analyse Katas K1 / SA",
    layout="wide",
)


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
    df = pd.read_csv(csv_path, sep=None, engine='python', encoding='utf-8')

    # Normalisation de la colonne Note (virgule -> point si besoin)
    if 'Note' in df.columns and df['Note'].dtype == object:
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

    # ── Display formatting (visual only, automatic for all tabs) ──
    if 'Nom' in df.columns:
        df['Nom'] = df['Nom'].astype(str).apply(fmt_athlete).astype('category')
    if 'Kata' in df.columns:
        df['Kata'] = df['Kata'].astype(str).apply(fmt_underscore).astype('category')
    if 'Competition' in df.columns:
        df['Competition'] = df['Competition'].astype(str).apply(fmt_underscore).astype('category')
    if 'Style' in df.columns:
        df['Style'] = df['Style'].astype(str).apply(fmt_underscore).astype('category')

    return df


def main():
    st.title("Suivi & Analyse des Katas – Premier League (K1) & Series A (SA)")

    # Chargement des données
    if not DATA_PATH.exists():
        st.error(f"Fichier de données introuvable : {DATA_PATH}")
        st.stop()

    data = load_data(DATA_PATH)

    # Onglets principaux
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
        "Dataset",
        "Focus Athlète",
        "Progression temporelle",
        "Score différentiel",
        "Analyse continentale",
        "Avancement par tour",
        "Diversité kata",
        "Graphiques interactifs",
        "ACM",
        "Probabilité de victoire",
    ])

    with tab1:
        show_dataset_tab(data)

    with tab2:
        show_athlete_focus_tab(data)

    with tab3:
        show_progression_tab(data)

    with tab4:
        show_score_differential_tab(data)

    with tab5:
        show_continental_tab(data)

    with tab6:
        show_tour_advancement_tab(data)

    with tab7:
        show_kata_diversity_tab(data)

    with tab8:
        show_graphs_tab(data)

    with tab9:
        show_acm_tab(data)

    with tab10:
        show_proba_victoire_kata_tab(data)

    # Glossaire dans la sidebar
    show_glossaire()

    # Footer global
    add_footer()


if __name__ == "__main__":
    main()
