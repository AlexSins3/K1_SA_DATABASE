# tabs/proba_score.py — Prédiction de score aux drapeaux
"""
Onglet "Probabilité de score" : prédit la distribution des scores
aux drapeaux (ex: 7-0, 6-1, 5-2, 4-3 en K1 ; 5-0, 4-1, 3-2 en SA).

Utilise le meilleur modèle par format :
- K1 : LightGBM (Acc±1 = 69%)
- SA : Ordinal optimisé (Acc±1 = 65%)
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.data_helpers import safe_mode
from utils.display import fmt_tour
from utils.lang import t, get_lang
from utils.ui import filter_panel_open, filter_panel_close


# ═══════════════════════════════════════════════════════════════════════════════
# Paths & Constants
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_DIR = Path(__file__).resolve().parent.parent / "model" / "proba_victory" / "results"

TOUR_ORDER = {
    "Pool_1": 1, "Pool_2": 2, "Pool_3": 3,
    "T1": 1, "T2": 2, "T3": 3,
    "PW1": 4, "PW2": 5, "PW3": 6,
    "RP1": 4, "RP2": 5, "RP3": 6, "RP4": 7,
    "R1": 7, "R2": 8,
    "Bronze": 9, "Final": 10,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Model Loading (cached)
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def _load_score_models():
    """Charge les modèles de score. Retourne dict {K1: ..., SA: ...}."""
    models = {}

    # K1 → LightGBM (meilleur Acc±1)
    lgbm_path = MODEL_DIR / "lgbm_models.pkl"
    if lgbm_path.exists():
        with open(lgbm_path, "rb") as f:
            lgbm_data = pickle.load(f)
        if "K1" in lgbm_data:
            models["K1"] = {
                "model": lgbm_data["K1"]["model"],
                "type": "lgbm",
                "features": lgbm_data["K1"]["model"].feature_name_,
                "accuracy_pm1": 0.69,
            }

    # SA → Ordinal (meilleur Acc±1)
    ordinal_path = MODEL_DIR / "ordinal_models.pkl"
    if ordinal_path.exists():
        with open(ordinal_path, "rb") as f:
            ordinal_data = pickle.load(f)
        if "SA" in ordinal_data:
            models["SA"] = {
                "model": ordinal_data["SA"]["model"],
                "scaler": ordinal_data["SA"]["scaler"],
                "type": "ordinal",
                "features": ordinal_data["SA"].get("selected_features", []),
                "accuracy_pm1": 0.65,
            }

    # Transfer model (pour compute Transfer_Score)
    transfer_path = MODEL_DIR / "transfer_model.pkl"
    if transfer_path.exists():
        with open(transfer_path, "rb") as f:
            transfer_data = pickle.load(f)
        models["_transfer"] = transfer_data

    return models


# ═══════════════════════════════════════════════════════════════════════════════
# Feature computation for a single pair (from app data)
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_pair_features(
    df: pd.DataFrame,
    nom_a: str, nom_b: str,
    type_compet: str, n_tour: str,
) -> Optional[pd.DataFrame]:
    """
    Calcule les features pour une paire A vs B à partir des données complètes.
    Retourne un DataFrame d'une ligne avec toutes les features nécessaires.
    """
    # Filtrer les données historiques (tout sauf le match prédit)
    hist = df.copy()

    # Données athlètes
    a_data = hist[hist["Nom"] == nom_a]
    b_data = hist[hist["Nom"] == nom_b]

    if a_data.empty or b_data.empty:
        return None

    # ── Features de ranking ──
    a_ranking = pd.to_numeric(a_data["Ranking"], errors="coerce").dropna()
    b_ranking = pd.to_numeric(b_data["Ranking"], errors="coerce").dropna()
    a_rank_last = float(a_ranking.iloc[-1]) if len(a_ranking) > 0 else 50.0
    b_rank_last = float(b_ranking.iloc[-1]) if len(b_ranking) > 0 else 50.0

    diff_ranking = b_rank_last - a_rank_last
    log_ranking_ratio = np.log1p(b_rank_last) - np.log1p(a_rank_last)

    # ── Features de performance ──
    a_victoire = a_data["Victoire"].astype(str).str.lower().isin(["true", "1", "vrai", "yes"])
    b_victoire = b_data["Victoire"].astype(str).str.lower().isin(["true", "1", "vrai", "yes"])
    a_wins = int(a_victoire.sum())
    a_total = len(a_data)
    b_wins = int(b_victoire.sum())
    b_total = len(b_data)

    a_winrate = (a_wins + 2) / (a_total + 4)  # Bayesian smoothing
    b_winrate = (b_wins + 2) / (b_total + 4)
    diff_winrate = a_winrate - b_winrate

    # Recent form (last 10 matches)
    a_recent = float(a_victoire.tail(10).mean()) if len(a_data) >= 3 else 0.5
    b_recent = float(b_victoire.tail(10).mean()) if len(b_data) >= 3 else 0.5
    diff_form = a_recent - b_recent

    # ── Notes ──
    a_notes = pd.to_numeric(a_data["Note"], errors="coerce").dropna()
    b_notes = pd.to_numeric(b_data["Note"], errors="coerce").dropna()
    a_note_mean = float(a_notes.mean()) if len(a_notes) > 0 else 24.0
    b_note_mean = float(b_notes.mean()) if len(b_notes) > 0 else 24.0
    diff_note_mean = a_note_mean - b_note_mean

    a_note_std = float(a_notes.std()) if len(a_notes) > 2 else 1.0
    b_note_std = float(b_notes.std()) if len(b_notes) > 2 else 1.0
    diff_note_std = a_note_std - b_note_std

    # Note trend (dernier tiers vs premier tiers)
    def _note_trend(notes):
        if len(notes) < 4:
            return 0.0
        third = len(notes) // 3
        return float(notes.iloc[-third:].mean() - notes.iloc[:third].mean())

    diff_note_trend = _note_trend(a_notes) - _note_trend(b_notes)

    # ── Expérience ──
    exp_diff = np.log1p(a_total) - np.log1p(b_total)

    # ── Kata-specific ──
    # On prend le kata le plus récent de chaque athlète pour le calcul
    a_kata_data = a_data  # Tous les katas de A
    b_kata_data = b_data
    a_kata_wins = int(a_victoire.sum())
    a_kata_total = len(a_kata_data)
    b_kata_wins = int(b_victoire.sum())
    b_kata_total = len(b_kata_data)
    a_kata_wr = (a_kata_wins + 1) / (a_kata_total + 2)
    b_kata_wr = (b_kata_wins + 1) / (b_kata_total + 2)
    diff_kata_wr = a_kata_wr - b_kata_wr

    # Kata diversity
    a_diversity = a_data["Kata"].nunique() if "Kata" in a_data.columns else 1
    b_diversity = b_data["Kata"].nunique() if "Kata" in b_data.columns else 1
    diff_kata_diversity = a_diversity - b_diversity

    # Kata global WR
    diff_kata_global_wr = 0.0  # Simplified

    # A Kata Note Centered
    a_kata_note_centered = a_note_mean - b_note_mean  # Approximation

    # ── H2H ──
    # Matchs entre A et B
    h2h_a = hist[(hist["Nom"] == nom_a)]
    # On cherche les matchs où ils se sont affrontés
    a_h2h_rate = 0.5  # Default

    # ── Age ──
    a_age = pd.to_numeric(a_data["Age"], errors="coerce").dropna()
    b_age = pd.to_numeric(b_data["Age"], errors="coerce").dropna()
    a_age_val = float(a_age.iloc[-1]) if len(a_age) > 0 else 25.0
    b_age_val = float(b_age.iloc[-1]) if len(b_age) > 0 else 25.0
    diff_age = (a_age_val - b_age_val) / 10.0

    # ── Contexte ──
    tour_rank = TOUR_ORDER.get(str(n_tour), 5) / 10.0
    is_male = 1 if safe_mode(a_data["Sexe"], default="M") == "M" else 0
    same_nation = int(safe_mode(a_data["Nation"], "") == safe_mode(b_data["Nation"], ""))
    same_continent = int(safe_mode(a_data["Continent"], "") == safe_mode(b_data["Continent"], ""))
    same_style = int(safe_mode(a_data["Style"], "") == safe_mode(b_data["Style"], ""))
    home_advantage = 0  # Simplified (no competition region info at prediction time)

    # ── Drapeaux ──
    a_flags = pd.to_numeric(a_data["Drapeau"], errors="coerce").dropna()
    b_flags = pd.to_numeric(b_data["Drapeau"], errors="coerce").dropna()
    a_flag_mean = float(a_flags.mean()) if len(a_flags) > 0 else 0.0
    b_flag_mean = float(b_flags.mean()) if len(b_flags) > 0 else 0.0
    diff_flag_mean = a_flag_mean - b_flag_mean

    # ── Interactions ──
    ranking_x_ismale = log_ranking_ratio * is_male
    winrate_x_ismale = diff_winrate * is_male
    ranking_x_tour = log_ranking_ratio * tour_rank
    winrate_x_tour = diff_winrate * tour_rank
    form_x_tour = diff_form * tour_rank

    # ── Transfer Score ──
    transfer_score = 0.0
    models = _load_score_models()
    if "_transfer" in models:
        transfer_feats = {
            "Diff_Ranking": diff_ranking,
            "Log_Ranking_Ratio": log_ranking_ratio,
            "Diff_Age": diff_age,
            "Diff_Winrate": diff_winrate,
            "Diff_Form": diff_form,
            "Diff_Kata_WR": diff_kata_wr,
            "A_H2H_Rate": a_h2h_rate,
            "Exp_Diff": exp_diff,
            "Same_Nation": same_nation,
            "Same_Continent": same_continent,
            "Same_Style": same_style,
            "Tour_Rank": tour_rank,
            "Is_K1": 1 if type_compet == "K1" else 0,
            "Is_Male": is_male,
            "Home_Advantage": home_advantage,
            "Diff_Kata_Diversity": diff_kata_diversity,
            "Ranking_x_Tour": ranking_x_tour,
            "Ranking_x_IsMale": ranking_x_ismale,
        }
        tf_model = models["_transfer"]["model"]
        tf_feats = models["_transfer"]["features"]
        tf_X = pd.DataFrame([transfer_feats])[tf_feats]
        transfer_score = float(tf_model.predict(tf_X)[0])

    # ── Assembler toutes les features ──
    features = {
        "Diff_Ranking": diff_ranking,
        "Log_Ranking_Ratio": log_ranking_ratio,
        "Exp_Diff": exp_diff,
        "Diff_Winrate": diff_winrate,
        "Diff_Form": diff_form,
        "Diff_Note_Mean": diff_note_mean,
        "Diff_Note_Trend": diff_note_trend,
        "Diff_Kata_WR": diff_kata_wr,
        "Diff_Kata_Global_WR": diff_kata_global_wr,
        "Diff_Kata_Diversity": diff_kata_diversity,
        "A_Kata_Note_Centered": a_kata_note_centered,
        "A_H2H_Rate": a_h2h_rate,
        "Diff_Age": diff_age,
        "Tour_Rank": tour_rank,
        "Is_Male": is_male,
        "Same_Nation": same_nation,
        "Same_Continent": same_continent,
        "Same_Style": same_style,
        "Home_Advantage": home_advantage,
        "Diff_Note_Std": diff_note_std,
        "Diff_Flag_Mean": diff_flag_mean,
        "Transfer_Score": transfer_score,
        "Ranking_x_IsMale": ranking_x_ismale,
        "Winrate_x_IsMale": winrate_x_ismale,
        "Ranking_x_Tour": ranking_x_tour,
        "Winrate_x_Tour": winrate_x_tour,
        "Form_x_Tour": form_x_tour,
    }

    return pd.DataFrame([features])


def _predict_score_proba(features_df: pd.DataFrame, type_compet: str) -> Optional[np.ndarray]:
    """
    Prédit la distribution de probabilité sur les scores.
    Retourne un array de taille (max_flags+1,) avec les probas.
    """
    models = _load_score_models()
    if type_compet not in models:
        return None

    model_info = models[type_compet]
    feat_cols = model_info["features"]

    # S'assurer que toutes les features nécessaires sont présentes
    X = features_df[feat_cols].fillna(0).astype(float)

    if model_info["type"] == "lgbm":
        proba = model_info["model"].predict_proba(X)[0]
    elif model_info["type"] == "ordinal":
        X_scaled = model_info["scaler"].transform(X)
        # mord n'a pas predict_proba, on utilise une approximation
        # On va plutôt utiliser la prédiction ponctuelle + distribution gaussienne
        pred = model_info["model"].predict(X_scaled)[0]
        max_flags = 5  # SA
        # Créer une distribution approximative centrée sur la prédiction
        proba = np.zeros(max_flags + 1)
        for i in range(max_flags + 1):
            # Distribution triangulaire autour de la prédiction
            dist = abs(i - pred)
            proba[i] = max(0, 1.0 - 0.35 * dist)
        proba = proba / proba.sum()
    else:
        return None

    return proba


# ═══════════════════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════════════════

def show_proba_score_tab(data: pd.DataFrame) -> None:
    """Onglet principal de prédiction de score aux drapeaux."""

    st.header(t("Prédiction du score aux drapeaux"))

    if get_lang() == "en":
        st.markdown(
            """
This model predicts the **probable flag score** of a match (e.g. 5-2, 7-0, 3-2...).

- 🎯 **Accuracy**: predictions are correct **±1 flag** in ~67% of cases
- 📊 The chart shows the probability of each possible score
- 🏆 The predicted winner is the one with the most flags
            """
        )
    else:
        st.markdown(
            """
Ce modèle prédit le **score probable aux drapeaux** d'un match (ex: 5-2, 7-0, 3-2...).

- 🎯 **Précision** : les prédictions sont correctes à **±1 drapeau** dans ~67% des cas
- 📊 Le graphique montre la probabilité de chaque score possible
- 🏆 Le vainqueur prédit est celui qui obtient le plus de drapeaux
            """
        )

    models = _load_score_models()
    available_types = [t for t in ["K1", "SA"] if t in models]

    if not available_types:
        st.error(t("Aucun modèle de score disponible. Lancez d'abord l'entraînement."))
        return

    df = data.copy()
    for col in ["Year", "Note", "Ranking", "Age", "Drapeau"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Layout ──
    filters_col, content_col = st.columns([0.9, 2.4])

    with filters_col:
        filter_panel_open()
        st.markdown("### " + t("Paramètres"))

        # Type de compétition
        type_labels = {"K1": "Premier League (K1) – 7 juges", "SA": "Series A (SA) – 5 juges"}
        type_compet = st.radio(
            t("Format"),
            available_types,
            format_func=lambda x: type_labels.get(x, x),
            key="score_type_compet",
        )

        max_flags = 7 if type_compet == "K1" else 5

        # Filtrer par type
        df_scope = df[df["Type_Compet"] == type_compet].copy() if "Type_Compet" in df.columns else df.copy()

        # Sélection des athlètes
        athlete_names = sorted(df_scope["Nom"].dropna().unique().tolist())
        if not athlete_names:
            st.warning(t("Aucun athlète disponible."))
            filter_panel_close()
            return

        nom_a = st.selectbox(t("Athlète A (aka)"), athlete_names, key="score_nom_a")

        # Filtrer par sexe (même sexe seulement)
        sexe_a = safe_mode(df_scope[df_scope["Nom"] == nom_a]["Sexe"], default=None)
        if sexe_a is not None:
            b_candidates = sorted(
                df_scope[(df_scope["Sexe"] == sexe_a) & (df_scope["Nom"] != nom_a)]["Nom"]
                .dropna().unique().tolist()
            )
        else:
            b_candidates = sorted(df_scope[df_scope["Nom"] != nom_a]["Nom"].dropna().unique().tolist())

        if not b_candidates:
            st.warning(t("Aucun adversaire compatible trouvé."))
            filter_panel_close()
            return

        nom_b = st.selectbox(t("Athlète B (shiro)"), b_candidates, key="score_nom_b")

        # Tour
        tour_options = sorted(df_scope["N_Tour"].dropna().astype(str).unique().tolist())
        if tour_options:
            n_tour = st.selectbox(
                t("Tour simulé"),
                tour_options,
                format_func=fmt_tour,
                key="score_tour",
            )
        else:
            n_tour = "Pool_1"

        st.markdown("---")
        run = st.button(t("🎯 Prédire le score"), key="score_predict", type="primary")
        filter_panel_close()

    with content_col:
        if not run:
            st.info(t("Sélectionnez deux athlètes et cliquez sur 'Prédire le score'."))
            return

        # Compute features
        with st.spinner(t("Calcul en cours...")):
            features_df = _compute_pair_features(df, nom_a, nom_b, type_compet, n_tour)

        if features_df is None:
            st.error(t("Données insuffisantes pour cette paire d'athlètes."))
            return

        # Predict
        proba = _predict_score_proba(features_df, type_compet)
        if proba is None:
            st.error(t("Modèle non disponible pour ce format."))
            return

        predicted_score_a = int(np.argmax(proba))
        predicted_score_b = max_flags - predicted_score_a

        # ── Résultats principaux ──
        st.subheader(f"{nom_a}  vs  {nom_b}")
        st.caption(f"{type_labels[type_compet]} · {fmt_tour(n_tour)}")

        # Score prédit
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.metric(
                nom_a,
                f"{predicted_score_a} 🚩",
                help=t("Nombre de drapeaux prédit pour A"),
            )
        with col2:
            st.markdown(
                f"<div style='text-align:center; font-size:2em; padding-top:20px;'>—</div>",
                unsafe_allow_html=True,
            )
        with col3:
            st.metric(
                nom_b,
                f"{predicted_score_b} 🚩",
                help=t("Nombre de drapeaux prédit pour B"),
            )

        # Vainqueur
        if predicted_score_a > predicted_score_b:
            winner = nom_a
            color = "#28a745"
        elif predicted_score_b > predicted_score_a:
            winner = nom_b
            color = "#dc3545"
        else:
            winner = t("Indéterminé")
            color = "#ffc107"

        st.markdown(
            f"<div style='text-align:center; font-size:1.2em; color:{color}; "
            f"font-weight:bold; margin:10px 0;'>"
            f"🏆 {t('Vainqueur prédit')} : {winner} ({predicted_score_a}-{predicted_score_b})"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── Graphique distribution ──
        st.markdown(f"#### {t('Distribution des scores probables')}")

        scores_labels = [f"{i}-{max_flags - i}" for i in range(max_flags + 1)]
        colors = []
        for i in range(max_flags + 1):
            if i > max_flags - i:
                colors.append("#28a745")  # A gagne
            elif i < max_flags - i:
                colors.append("#dc3545")  # B gagne
            else:
                colors.append("#ffc107")  # Nul (impossible en réalité)

        fig = go.Figure(data=[
            go.Bar(
                x=scores_labels,
                y=proba * 100,
                marker_color=colors,
                text=[f"{p:.0f}%" for p in proba * 100],
                textposition="outside",
                hovertemplate="%{x}<br>Probabilité: %{y:.1f}%<extra></extra>",
            )
        ])
        fig.update_layout(
            xaxis_title=t("Score (A - B)"),
            yaxis_title=t("Probabilité (%)"),
            yaxis_range=[0, max(proba * 100) * 1.3],
            showlegend=False,
            height=350,
            margin=dict(t=20, b=40),
        )
        # Add legend annotations
        fig.add_annotation(
            x=0.02, y=0.95, xref="paper", yref="paper",
            text=f"🟢 {nom_a} {t('gagne')}",
            showarrow=False, font=dict(size=11),
        )
        fig.add_annotation(
            x=0.98, y=0.95, xref="paper", yref="paper",
            text=f"🔴 {nom_b} {t('gagne')}",
            showarrow=False, font=dict(size=11), xanchor="right",
        )
        st.plotly_chart(fig, use_container_width=True, key="score_proba_chart")

        # ── Probabilité de victoire agrégée ──
        prob_a_wins = sum(proba[i] for i in range(max_flags + 1) if i > max_flags - i)
        prob_b_wins = sum(proba[i] for i in range(max_flags + 1) if i < max_flags - i)

        st.markdown(f"#### {t('Probabilité de victoire')}")
        col_pa, col_pb = st.columns(2)
        with col_pa:
            st.metric(nom_a, f"{prob_a_wins * 100:.0f}%")
        with col_pb:
            st.metric(nom_b, f"{prob_b_wins * 100:.0f}%")

        # ── Info modèle ──
        model_info = models[type_compet]
        acc_pm1 = model_info["accuracy_pm1"]
        model_type_label = "LightGBM" if model_info["type"] == "lgbm" else "Logistique Ordinale"

        st.markdown("---")
        st.caption(
            f"ℹ️ {t('Modèle')} : {model_type_label} · "
            f"{t('Précision')} ±1 {t('drapeau')} : **{acc_pm1 * 100:.0f}%** · "
            f"{max_flags} {t('juges')}"
        )
        if get_lang() == "en":
            st.caption(
                "⚠️ These predictions are probabilities. "
                "The actual result depends on kata choice, day form, and judges."
            )
        else:
            st.caption(
                "⚠️ Ces prédictions sont des probabilités. "
                "Le résultat réel dépend du kata choisi, de la forme du jour et des juges."
            )
