# app.py

from pathlib import Path

import pandas as pd
import streamlit as st

from config import DATA_PATH
from utils.ui import add_footer
from utils.interpretations import show_glossaire
from utils.display import fmt_athlete, fmt_underscore
from utils.lang import render_language_selector, t
from tabs.dataset_view import show_dataset_tab
from tabs.athlete_focus import show_athlete_focus_tab
from tabs.graphs import show_graphs_tab
from tabs.acm import show_acm_tab
from tabs.proba_victoire_kata import show_proba_victoire_kata_tab
from tabs.kata_comparison import show_kata_comparison_tab
from tabs.match_analysis import show_match_analysis_tab
from tabs.tendances import show_tendances_tab


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
    df = pd.read_csv(csv_path, sep=None, engine='python', encoding='utf-8-sig')

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

    # Conversion de la colonne Drapeau (système de drapeaux 2026+)
    if 'Drapeau' in df.columns:
        df['Drapeau'] = pd.to_numeric(df['Drapeau'], errors='coerce')

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
    # ── Language selector (top-right) ──
    render_language_selector()

    st.title(t("Suivi & Analyse des Katas – Premier League (K1) & Series A (SA)"))

    # Chargement des données
    if not DATA_PATH.exists():
        st.error(f"Fichier de données introuvable : {DATA_PATH}")
        st.stop()

    data = load_data(DATA_PATH)

    # Onglets principaux — organisés par utilité
    # Tier 1 : Préparation compétition (coach)
    # Tier 2 : Analyse approfondie (coach curieux / data analyst)
    # Tier 3 : Outils

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "🥋 " + t("Focus Athlète"),
        "🎯 " + t("Probabilité de victoire"),
        "🆚 " + t("Comparaison de katas"),
        "🔍 " + t("Tendances & Réalités"),
        "⚔️ " + t("Analyse des matchs"),
        "🗺️ " + t("ACM"),
        "📊 " + t("Exploration libre"),
        "📋 " + t("Dataset"),
    ])

    with tab1:
        show_athlete_focus_tab(data)

    with tab2:
        show_proba_victoire_kata_tab(data)

    with tab3:
        show_kata_comparison_tab(data)

    with tab4:
        show_tendances_tab(data)

    with tab5:
        show_match_analysis_tab(data)

    with tab6:
        show_acm_tab(data)

    with tab7:
        show_graphs_tab(data)

    with tab8:
        show_dataset_tab(data)

    # Glossaire dans la sidebar
    show_glossaire()

    # Footer global
    add_footer()


if __name__ == "__main__":
    main()
