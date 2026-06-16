# tabs/proba_victoire_kata_v5.py — v5 : LightGBM Optuna pré-entraîné
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.ui import filter_panel_open, filter_panel_close
from utils.data_helpers import safe_mode
from utils.interpretations import show_tab_help, proba_bar_html, proba_color, _color_badge
from utils.display import fmt_tour, format_display_df
from utils.lang import t, get_lang


# ═══════════════════════════════════════════════════════════════════════════════
# Paths & Constants
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_DIR = Path(__file__).resolve().parent.parent / "model" / "proba_kata_victory" / "results"
TRANSFER_DIR = Path(__file__).resolve().parent.parent / "model" / "proba_victory" / "results"

TOUR_ORDER = {
    "Pool_1": 1, "Pool_2": 2, "Pool_3": 3, "Pool_4": 4,
    "T1": 1, "T2": 2, "T3": 3,
    "PW1": 4, "PW2": 5, "PW3": 6,
    "RP1": 4, "RP2": 5, "RP3": 6, "RP4": 7,
    "R1": 7, "R2": 8,
    "Bronze": 9, "Final": 10,
}

_TOUR_RANK_MAP = {
    "pool_1": 1, "pool_2": 2, "pool_3": 3, "pool_4": 4,
    "round_1": 2, "round_2": 3, "1/8": 3, "1/4": 4,
    "rp1": 3, "rp2": 3, "rp3": 4, "rp4": 4,
    "bronze": 5, "finale": 6, "final": 6,
}

# 34 features expected by the model (same order as training)
MODEL_FEATURES = [
    "WinRate_Adv", "Ranking_Adv", "Exp_Diff", "Momentum_Diff", "Recent_5_Diff",
    "Note_Trend_Diff", "Note_Std_Diff", "Age_Diff", "Flag_WR_Diff", "Win_Streak_Diff",
    "A_Kata_WR", "Log_A_Kata_N", "A_Kata_Note_C", "Note_Diff",
    "A_Kata_Consistency", "Is_Fav_Kata_A",
    "H2H_Adv", "B_Weakness_C", "B_Seen_Kata", "Opp_Style_WR",
    "Kata_Tour_C", "Kata_Effect",
    "Same_Nation", "Same_Style", "Is_K1", "Is_Male", "Is_Red",
    "Is_Home", "Tour_Rank",
    "Transfer_Score",
    "Ranking_x_Tour", "WinRate_x_Tour", "KataWR_x_Tour",
    "Kata_Diversity_Diff",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Small helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _to_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def _center_rate(x: float) -> float:
    try:
        return float(x) - 0.5
    except Exception:
        return 0.0


def _tour_to_rank(tour_str: str) -> float:
    return _TOUR_RANK_MAP.get(str(tour_str).strip().lower(), 2.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Model loading (cached)
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def _load_kata_victory_model():
    """Charge le modèle LightGBM Optuna pré-entraîné."""
    model_path = MODEL_DIR / "best_model.pkl"
    if not model_path.exists():
        return None, None
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["features"]


@st.cache_resource
def _load_transfer_model():
    """Charge le modèle de transfert (GradientBoosting sur notes → dominance)."""
    path = TRANSFER_DIR / "transfer_model.pkl"
    if not path.exists():
        return None, None
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["features"]


# ═══════════════════════════════════════════════════════════════════════════════
# Pairing R/B → matches
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _build_paired_matches(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy().reset_index(drop=True)
    needed = ["Nom", "Ceinture", "Kata", "N_Tour", "Competition", "Type_Compet", "Victoire"]
    for c in needed:
        if c not in d.columns:
            raise ValueError(f"Colonne manquante: {c}")

    for col_default in ["Year", "Nation", "Style", "Sexe", "Note", "Ranking", "Age",
                        "Continent", "Region_monde", "Drapeau"]:
        if col_default not in d.columns:
            d[col_default] = np.nan

    d = d[d["Nom"].notna() & (d["Nom"].astype(str).str.strip() != "")].reset_index(drop=True)
    d = d.sort_values(["Competition", "Year", "Type_Compet", "N_Tour"], kind="mergesort").reset_index(drop=True)

    belt = d["Ceinture"].astype(str)
    belt_next = belt.shift(-1)
    comp = d["Competition"].astype(str)
    tc = d["Type_Compet"].astype(str)
    nt = d["N_Tour"].astype(str)
    yr = d["Year"].astype(str)

    mask = (
        belt.isin(["R", "B"])
        & belt_next.isin(["R", "B"])
        & (belt != belt_next)
        & (comp == comp.shift(-1))
        & (tc == tc.shift(-1))
        & (nt == nt.shift(-1))
        & (yr == yr.shift(-1))
    )

    idx = mask[mask].index.values
    if len(idx) == 0:
        return pd.DataFrame()

    r1 = d.loc[idx].reset_index(drop=True)
    r2 = d.loc[idx + 1].reset_index(drop=True)

    is_r1_red = r1["Ceinture"].astype(str).values == "R"

    def _pick(col):
        return np.where(is_r1_red, r1[col].values, r2[col].values)
    def _pick_inv(col):
        return np.where(is_r1_red, r2[col].values, r1[col].values)

    v1 = r1["Victoire"].astype(str).str.lower().isin(["true", "1", "vrai", "yes"]).astype(int).values
    v2 = r2["Victoire"].astype(str).str.lower().isin(["true", "1", "vrai", "yes"]).astype(int).values
    red_win = np.where(is_r1_red, v1, v2)

    def _notes(series):
        return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce").values

    result = pd.DataFrame({
        "Competition": _pick("Competition"),
        "Year": _pick("Year"),
        "Type_Compet": _pick("Type_Compet"),
        "N_Tour": _pick("N_Tour"),
        "Red_Nom": _pick("Nom"),
        "Blue_Nom": _pick_inv("Nom"),
        "Red_Kata": _pick("Kata"),
        "Blue_Kata": _pick_inv("Kata"),
        "Red_Nation": _pick("Nation"),
        "Blue_Nation": _pick_inv("Nation"),
        "Red_Style": _pick("Style"),
        "Blue_Style": _pick_inv("Style"),
        "Red_Sexe": _pick("Sexe"),
        "Blue_Sexe": _pick_inv("Sexe"),
        "Red_Note": np.where(is_r1_red, _notes(r1["Note"]), _notes(r2["Note"])),
        "Blue_Note": np.where(is_r1_red, _notes(r2["Note"]), _notes(r1["Note"])),
        "Red_Ranking": np.where(is_r1_red, pd.to_numeric(r1["Ranking"], errors="coerce").values,
                                pd.to_numeric(r2["Ranking"], errors="coerce").values),
        "Blue_Ranking": np.where(is_r1_red, pd.to_numeric(r2["Ranking"], errors="coerce").values,
                                 pd.to_numeric(r1["Ranking"], errors="coerce").values),
        "Red_Age": np.where(is_r1_red, pd.to_numeric(r1["Age"], errors="coerce").values,
                            pd.to_numeric(r2["Age"], errors="coerce").values),
        "Blue_Age": np.where(is_r1_red, pd.to_numeric(r2["Age"], errors="coerce").values,
                             pd.to_numeric(r1["Age"], errors="coerce").values),
        "Red_Continent": _pick("Continent"),
        "Blue_Continent": _pick_inv("Continent"),
        "Red_Region": _pick("Region_monde"),
        "Blue_Region": _pick_inv("Region_monde"),
        "Red_Drapeau": np.where(is_r1_red, pd.to_numeric(r1["Drapeau"], errors="coerce").values,
                                pd.to_numeric(r2["Drapeau"], errors="coerce").values),
        "Blue_Drapeau": np.where(is_r1_red, pd.to_numeric(r2["Drapeau"], errors="coerce").values,
                                 pd.to_numeric(r1["Drapeau"], errors="coerce").values),
        "Red_Win": red_win,
    })
    return result


def _to_directed_rows(matches: pd.DataFrame) -> pd.DataFrame:
    if matches.empty:
        return pd.DataFrame()

    base_cols = ["N_Tour", "Competition", "Year", "Type_Compet"]

    red = matches[base_cols].copy()
    red["Nom"] = matches["Red_Nom"]
    red["Opponent"] = matches["Blue_Nom"]
    red["Kata"] = matches["Red_Kata"]
    red["Opp_Kata"] = matches["Blue_Kata"]
    red["Nation"] = matches["Red_Nation"]
    red["Opp_Nation"] = matches["Blue_Nation"]
    red["Style"] = matches["Red_Style"]
    red["Opp_Style"] = matches["Blue_Style"]
    red["Sexe"] = matches["Red_Sexe"]
    red["Note"] = matches["Red_Note"]
    red["Opp_Note"] = matches["Blue_Note"]
    red["Ranking"] = matches["Red_Ranking"]
    red["Opp_Ranking"] = matches["Blue_Ranking"]
    red["Age"] = matches["Red_Age"]
    red["Opp_Age"] = matches["Blue_Age"]
    red["Continent"] = matches["Red_Continent"]
    red["Opp_Continent"] = matches["Blue_Continent"]
    red["Region"] = matches["Red_Region"]
    red["Drapeau"] = matches["Red_Drapeau"]
    red["Athlete_Win"] = matches["Red_Win"].astype(int)
    red["Is_Red"] = 1

    blue = matches[base_cols].copy()
    blue["Nom"] = matches["Blue_Nom"]
    blue["Opponent"] = matches["Red_Nom"]
    blue["Kata"] = matches["Blue_Kata"]
    blue["Opp_Kata"] = matches["Red_Kata"]
    blue["Nation"] = matches["Blue_Nation"]
    blue["Opp_Nation"] = matches["Red_Nation"]
    blue["Style"] = matches["Blue_Style"]
    blue["Opp_Style"] = matches["Red_Style"]
    blue["Sexe"] = matches["Blue_Sexe"]
    blue["Note"] = matches["Blue_Note"]
    blue["Opp_Note"] = matches["Red_Note"]
    blue["Ranking"] = matches["Blue_Ranking"]
    blue["Opp_Ranking"] = matches["Red_Ranking"]
    blue["Age"] = matches["Blue_Age"]
    blue["Opp_Age"] = matches["Red_Age"]
    blue["Continent"] = matches["Blue_Continent"]
    blue["Opp_Continent"] = matches["Red_Continent"]
    blue["Region"] = matches["Blue_Region"]
    blue["Drapeau"] = matches["Blue_Drapeau"]
    blue["Athlete_Win"] = (1 - matches["Red_Win"]).astype(int)
    blue["Is_Red"] = 0

    return pd.concat([red, blue], ignore_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Aggregates (computed from directed rows — same as before)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Aggregates:
    athlete_stats: pd.DataFrame
    athlete_kata: pd.DataFrame
    athlete_oppkata_losses: pd.DataFrame
    kata_tour: pd.DataFrame
    h2h: pd.DataFrame
    kata_effect: pd.DataFrame
    global_note_mean: float
    athlete_trend: pd.DataFrame
    athlete_recent: pd.DataFrame
    athlete_fav_kata: pd.DataFrame
    athlete_kata_diversity: pd.DataFrame
    opponent_seen_kata: pd.DataFrame
    athlete_flag_stats: pd.DataFrame
    athlete_recent_5: pd.DataFrame
    athlete_streak: pd.DataFrame
    # v5 — style WR + kata consistency
    style_wr: dict
    athlete_kata_consistency: pd.DataFrame


def _compute_aggregates(directed: pd.DataFrame) -> Aggregates:
    if directed.empty:
        empty = pd.DataFrame()
        return Aggregates(empty, empty, empty, empty, empty, empty, 0.0,
                          empty, empty, empty, empty, empty,
                          empty, empty, empty, {}, empty)

    d = directed.copy()
    global_note_mean = float(d["Note"].dropna().mean()) if d["Note"].notna().any() else 0.0

    # Athlete-level stats
    a = d.groupby("Nom").agg(
        Wins=("Athlete_Win", "sum"),
        Total=("Athlete_Win", "count"),
        Note_Mean=("Note", "mean"),
        Ranking_Mean=("Ranking", "mean"),
    ).reset_index()
    a["WinRate_Smoothed"] = (a["Wins"] + 2) / (a["Total"] + 4)

    # Athlete × Kata
    ak = d.groupby(["Nom", "Kata"])["Athlete_Win"].agg(["sum", "count"]).reset_index()
    ak.rename(columns={"sum": "Kata_Wins", "count": "Kata_Total"}, inplace=True)
    notes_gp = d.dropna(subset=["Note"]).groupby(["Nom", "Kata"])["Note"].agg(["mean", "std"]).reset_index()
    notes_gp.rename(columns={"mean": "Kata_Note_Mean", "std": "Kata_Note_Std"}, inplace=True)
    ak = ak.merge(notes_gp, on=["Nom", "Kata"], how="left")
    ak["Kata_WinRate_Smoothed"] = (ak["Kata_Wins"] + 3.0) / (ak["Kata_Total"] + 6.0)

    # Opponent weakness vs kata
    d["Loss"] = 1 - d["Athlete_Win"]
    ok = d.groupby(["Nom", "Opp_Kata"])["Loss"].agg(["sum", "count"]).reset_index()
    ok.rename(columns={"sum": "Losses_vs_Kata", "count": "Total_vs_Kata"}, inplace=True)
    ok["LossRate_vs_Kata_Smoothed"] = (ok["Losses_vs_Kata"] + 3.0) / (ok["Total_vs_Kata"] + 6.0)

    # Kata × Tour
    kt = d.groupby(["Kata", "N_Tour"])["Athlete_Win"].agg(["sum", "count"]).reset_index()
    kt.rename(columns={"sum": "Wins", "count": "Total"}, inplace=True)
    kt["Kata_Tour_WinRate_Smoothed"] = (kt["Wins"] + 2) / (kt["Total"] + 4)

    # Head-to-head
    tmp = d[["Nom", "Opponent", "Athlete_Win"]].copy()
    nom_str = tmp["Nom"].astype(str)
    opp_str = tmp["Opponent"].astype(str)
    tmp["A"] = np.where(nom_str <= opp_str, nom_str, opp_str)
    tmp["B"] = np.where(nom_str > opp_str, nom_str, opp_str)
    tmp["A_is_Nom"] = nom_str == tmp["A"]
    tmp["Win_for_A"] = np.where(tmp["A_is_Nom"], tmp["Athlete_Win"], 1 - tmp["Athlete_Win"])
    h2h = tmp.groupby(["A", "B"])["Win_for_A"].agg(["sum", "count"]).reset_index()
    h2h.rename(columns={"sum": "Wins_A", "count": "Total"}, inplace=True)
    h2h["WinRate_A_Smoothed"] = (h2h["Wins_A"] + 1.5) / (h2h["Total"] + 3.0)

    # Kata residual effect
    a_map = a.set_index("Nom")["WinRate_Smoothed"].to_dict()
    d["Athlete_Base"] = d["Nom"].map(a_map).fillna(0.5)
    d["Residual_vs_Base"] = d["Athlete_Win"] - d["Athlete_Base"]
    ke = d.groupby("Kata")["Residual_vs_Base"].agg(["mean", "count"]).reset_index()
    ke.rename(columns={"mean": "Residual_Mean", "count": "Kata_Uses"}, inplace=True)
    k = 25.0
    ke["Kata_Effect"] = ke["Residual_Mean"] * (ke["Kata_Uses"] / (ke["Kata_Uses"] + k))

    # Note trend + std
    d_notes = d.dropna(subset=["Note"]).copy()
    d_notes["_match_idx"] = d_notes.groupby("Nom").cumcount()
    g_trend = d_notes.groupby("Nom").agg(
        _n=("_match_idx", "count"),
        _mean_x=("_match_idx", "mean"),
        _mean_y=("Note", "mean"),
        _var_x=("_match_idx", "var"),
        Note_Std=("Note", "std"),
    ).reset_index()
    d_notes["_xy"] = d_notes["_match_idx"] * d_notes["Note"]
    cov_xy = d_notes.groupby("Nom")["_xy"].mean().reset_index().rename(columns={"_xy": "_mean_xy"})
    g_trend = g_trend.merge(cov_xy, on="Nom", how="left")
    g_trend["_cov"] = g_trend["_mean_xy"] - g_trend["_mean_x"] * g_trend["_mean_y"]
    g_trend["Note_Trend"] = np.where(
        (g_trend["_var_x"] > 0) & (g_trend["_n"] >= 3),
        g_trend["_cov"] / g_trend["_var_x"],
        0.0,
    )
    g_trend["Note_Std"] = g_trend["Note_Std"].fillna(0.0)
    athlete_trend = g_trend[["Nom", "Note_Trend", "Note_Std"]].copy()

    # Recent WR (last 15)
    d_recent = d.copy()
    d_recent["_rev_idx"] = d_recent.groupby("Nom").cumcount(ascending=False)
    recent_15 = d_recent[d_recent["_rev_idx"] < 15]
    r_wr = recent_15.groupby("Nom")["Athlete_Win"].agg(["sum", "count"]).reset_index()
    r_wr.rename(columns={"sum": "Recent_Wins", "count": "Recent_Total"}, inplace=True)
    r_wr["Recent_WinRate"] = (r_wr["Recent_Wins"] + 2) / (r_wr["Recent_Total"] + 4)
    athlete_recent = r_wr[["Nom", "Recent_WinRate"]].copy()

    # Favourite kata
    fav = ak.sort_values("Kata_Total", ascending=False).drop_duplicates("Nom", keep="first")
    athlete_fav_kata = fav[["Nom", "Kata"]].rename(columns={"Kata": "Fav_Kata"}).copy()

    # Kata diversity
    kata_div = d.groupby("Nom").agg(
        Nb_Katas=("Kata", "nunique"),
        Nb_Matchs=("Athlete_Win", "count"),
    ).reset_index()
    kata_div["Kata_Diversity"] = kata_div["Nb_Katas"] / kata_div["Nb_Matchs"]
    athlete_kata_diversity = kata_div[["Nom", "Kata_Diversity"]].copy()

    # Opponent seen kata
    opp_seen = d.groupby(["Nom", "Opp_Kata"])["Athlete_Win"].count().reset_index()
    opp_seen.rename(columns={"Athlete_Win": "Seen_Count"}, inplace=True)

    # Flag-era WR
    d_flag = d[d["Note"].isna()].copy()
    if not d_flag.empty:
        flag_agg = d_flag.groupby("Nom")["Athlete_Win"].agg(["sum", "count"]).reset_index()
        flag_agg.rename(columns={"sum": "Flag_Wins", "count": "Flag_Total"}, inplace=True)
        flag_agg["Flag_WinRate"] = (flag_agg["Flag_Wins"] + 2) / (flag_agg["Flag_Total"] + 4)
        athlete_flag_stats = flag_agg[["Nom", "Flag_WinRate", "Flag_Total"]].copy()
    else:
        athlete_flag_stats = pd.DataFrame(columns=["Nom", "Flag_WinRate", "Flag_Total"])

    # Recent 5
    d_recent5 = d.copy()
    d_recent5["_rev_idx"] = d_recent5.groupby("Nom").cumcount(ascending=False)
    recent_5 = d_recent5[d_recent5["_rev_idx"] < 5]
    r5_wr = recent_5.groupby("Nom")["Athlete_Win"].agg(["sum", "count"]).reset_index()
    r5_wr.rename(columns={"sum": "Recent5_Wins", "count": "Recent5_Total"}, inplace=True)
    r5_wr["Recent5_WinRate"] = (r5_wr["Recent5_Wins"] + 1) / (r5_wr["Recent5_Total"] + 2)
    athlete_recent_5 = r5_wr[["Nom", "Recent5_WinRate"]].copy()

    # Win streak
    def _compute_streak(group):
        results = group["Athlete_Win"].values
        if len(results) == 0:
            return 0
        streak = 0
        last = results[-1]
        for r in reversed(results):
            if r == last:
                streak += 1
            else:
                break
        return streak if last == 1 else -streak

    streak_data = d.groupby("Nom").apply(_compute_streak, include_groups=False).reset_index()
    streak_data.columns = ["Nom", "Win_Streak"]
    athlete_streak = streak_data

    # Style WR (global)
    style_wr = {}
    for style in d["Style"].dropna().unique():
        s_data = d[d["Style"] == style]
        if len(s_data) >= 5:
            style_wr[str(style)] = float(s_data["Athlete_Win"].mean())
        else:
            style_wr[str(style)] = 0.5

    # Kata consistency per (athlete, kata) — std of notes
    athlete_kata_consistency = ak[["Nom", "Kata", "Kata_Note_Std"]].copy()
    athlete_kata_consistency["Kata_Note_Std"] = athlete_kata_consistency["Kata_Note_Std"].fillna(0.0)

    return Aggregates(
        a, ak, ok, kt, h2h, ke, global_note_mean,
        athlete_trend, athlete_recent, athlete_fav_kata,
        athlete_kata_diversity, opp_seen,
        athlete_flag_stats, athlete_recent_5, athlete_streak,
        style_wr, athlete_kata_consistency,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Compute aggregates (cached)
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _get_aggregates(df: pd.DataFrame) -> Tuple:
    """Build matches, directed rows, aggregates."""
    matches = _build_paired_matches(df)
    directed = _to_directed_rows(matches)
    ag = _compute_aggregates(directed)
    return matches, directed, ag


# ═══════════════════════════════════════════════════════════════════════════════
# Feature computation for prediction
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_base_features(df_scope, ag, matches_scope, selected_type_compet, nom_a, nom_b):
    """Compute pair-level features (independent of kata)."""
    nation_a = safe_mode(df_scope[df_scope["Nom"] == nom_a]["Nation"], default="")
    nation_b = safe_mode(df_scope[df_scope["Nom"] == nom_b]["Nation"], default="")
    same_nation = int(str(nation_a) == str(nation_b))

    style_a = safe_mode(df_scope[df_scope["Nom"] == nom_a]["Style"], default="")
    style_b = safe_mode(df_scope[df_scope["Nom"] == nom_b]["Style"], default="")
    same_style = int(str(style_a) == str(style_b))

    # WinRate + Ranking
    a_wr = b_wr = 0.5
    a_note_mean = b_note_mean = ag.global_note_mean
    a_ranking = b_ranking = 50.0
    if not ag.athlete_stats.empty:
        stats_map = ag.athlete_stats.set_index("Nom")
        if nom_a in stats_map.index:
            row_a = stats_map.loc[nom_a]
            a_wr = float(row_a["WinRate_Smoothed"])
            a_note_mean = float(row_a["Note_Mean"]) if pd.notna(row_a["Note_Mean"]) else ag.global_note_mean
            a_ranking = float(row_a["Ranking_Mean"]) if pd.notna(row_a["Ranking_Mean"]) else 50.0
        if nom_b in stats_map.index:
            row_b = stats_map.loc[nom_b]
            b_wr = float(row_b["WinRate_Smoothed"])
            b_note_mean = float(row_b["Note_Mean"]) if pd.notna(row_b["Note_Mean"]) else ag.global_note_mean
            b_ranking = float(row_b["Ranking_Mean"]) if pd.notna(row_b["Ranking_Mean"]) else 50.0

    winrate_adv = a_wr - b_wr
    ranking_adv = (b_ranking - a_ranking) / 100.0

    # H2H
    h2h_total = 0.0
    h2h_wr_a = 0.5
    if not ag.h2h.empty:
        a_key = min(str(nom_a), str(nom_b))
        b_key = max(str(nom_a), str(nom_b))
        h_index = ag.h2h.set_index(["A", "B"])
        if (a_key, b_key) in h_index.index:
            row = h_index.loc[(a_key, b_key)]
            h2h_total = float(row["Total"])
            wr_A = float(row["WinRate_A_Smoothed"])
            h2h_wr_a = wr_A if str(nom_a) == a_key else (1.0 - wr_A)
    h2h_adv = (h2h_wr_a - 0.5) * np.log1p(h2h_total)

    is_k1 = 1 if selected_type_compet == "Premier League (K1)" else (
        0 if selected_type_compet == "Series A (SA)" else 1
    )
    is_male = 1 if safe_mode(df_scope[df_scope["Nom"] == nom_a]["Sexe"], default="M") == "M" else 0

    # Match counts
    a_matches = b_matches = 0
    if not matches_scope.empty:
        a_matches = int(((matches_scope["Red_Nom"] == nom_a) | (matches_scope["Blue_Nom"] == nom_a)).sum())
        b_matches = int(((matches_scope["Red_Nom"] == nom_b) | (matches_scope["Blue_Nom"] == nom_b)).sum())

    # Note trend
    a_note_trend = b_note_trend = 0.0
    a_note_std = b_note_std = 0.0
    if not ag.athlete_trend.empty:
        t_map = ag.athlete_trend.set_index("Nom")
        if nom_a in t_map.index:
            a_note_trend = float(t_map.loc[nom_a, "Note_Trend"])
            a_note_std = float(t_map.loc[nom_a, "Note_Std"])
        if nom_b in t_map.index:
            b_note_trend = float(t_map.loc[nom_b, "Note_Trend"])
            b_note_std = float(t_map.loc[nom_b, "Note_Std"])

    # Momentum (recent 15)
    a_recent_wr = b_recent_wr = 0.5
    if not ag.athlete_recent.empty:
        r_map = ag.athlete_recent.set_index("Nom")
        if nom_a in r_map.index:
            a_recent_wr = float(r_map.loc[nom_a, "Recent_WinRate"])
        if nom_b in r_map.index:
            b_recent_wr = float(r_map.loc[nom_b, "Recent_WinRate"])

    # Age
    age_a = age_b = 25.0
    a_rows = df_scope[df_scope["Nom"] == nom_a]["Age"]
    b_rows = df_scope[df_scope["Nom"] == nom_b]["Age"]
    if a_rows.notna().any():
        age_a = float(pd.to_numeric(a_rows, errors="coerce").dropna().iloc[-1])
    if b_rows.notna().any():
        age_b = float(pd.to_numeric(b_rows, errors="coerce").dropna().iloc[-1])

    # Favourite kata
    a_fav_kata = None
    if not ag.athlete_fav_kata.empty:
        fav_map = ag.athlete_fav_kata.set_index("Nom")
        if nom_a in fav_map.index:
            a_fav_kata = str(fav_map.loc[nom_a, "Fav_Kata"])

    # Experience
    a_total_m = b_total_m = 1.0
    if not ag.athlete_stats.empty:
        s_map = ag.athlete_stats.set_index("Nom")
        if nom_a in s_map.index:
            a_total_m = float(s_map.loc[nom_a, "Total"])
        if nom_b in s_map.index:
            b_total_m = float(s_map.loc[nom_b, "Total"])

    # Kata diversity
    a_kata_div = b_kata_div = 0.5
    if not ag.athlete_kata_diversity.empty:
        d_map = ag.athlete_kata_diversity.set_index("Nom")
        if nom_a in d_map.index:
            a_kata_div = float(d_map.loc[nom_a, "Kata_Diversity"])
        if nom_b in d_map.index:
            b_kata_div = float(d_map.loc[nom_b, "Kata_Diversity"])

    # Continent / region (for Is_Home)
    continent_a = safe_mode(df_scope[df_scope["Nom"] == nom_a]["Continent"], default="")
    region_a = safe_mode(df_scope[df_scope["Nom"] == nom_a]["Region_monde"], default="")
    is_home = int(str(continent_a) == str(region_a))

    # Flag WR
    a_flag_wr = b_flag_wr = 0.5
    if not ag.athlete_flag_stats.empty:
        f_map = ag.athlete_flag_stats.set_index("Nom")
        if nom_a in f_map.index:
            a_flag_wr = float(f_map.loc[nom_a, "Flag_WinRate"])
        if nom_b in f_map.index:
            b_flag_wr = float(f_map.loc[nom_b, "Flag_WinRate"])

    # Recent 5
    a_recent5 = b_recent5 = 0.5
    if not ag.athlete_recent_5.empty:
        r5_map = ag.athlete_recent_5.set_index("Nom")
        if nom_a in r5_map.index:
            a_recent5 = float(r5_map.loc[nom_a, "Recent5_WinRate"])
        if nom_b in r5_map.index:
            b_recent5 = float(r5_map.loc[nom_b, "Recent5_WinRate"])

    # Win streak
    a_streak = b_streak = 0.0
    if not ag.athlete_streak.empty:
        s_map = ag.athlete_streak.set_index("Nom")
        if nom_a in s_map.index:
            a_streak = float(s_map.loc[nom_a, "Win_Streak"])
        if nom_b in s_map.index:
            b_streak = float(s_map.loc[nom_b, "Win_Streak"])

    # Opp style WR
    opp_style_wr = ag.style_wr.get(str(style_b), 0.5) - 0.5

    return {
        "same_nation": same_nation, "same_style": same_style,
        "is_k1": is_k1, "is_male": is_male, "is_home": is_home,
        "winrate_adv": winrate_adv, "ranking_adv": ranking_adv,
        "h2h_adv": h2h_adv, "h2h_wr_a": h2h_wr_a,
        "a_wr": a_wr, "b_wr": b_wr,
        "a_note_mean": a_note_mean, "b_note_mean": b_note_mean,
        "a_ranking": a_ranking, "b_ranking": b_ranking,
        "a_matches": a_matches, "b_matches": b_matches,
        "note_trend_diff": a_note_trend - b_note_trend,
        "note_std_diff": a_note_std - b_note_std,
        "momentum_diff": a_recent_wr - b_recent_wr,
        "age_diff": (age_a - age_b) / 10.0,
        "a_fav_kata": a_fav_kata,
        "exp_diff": float(np.log1p(a_total_m) - np.log1p(b_total_m)),
        "kata_diversity_diff": (a_kata_div - b_kata_div) / 5.0,
        "flag_wr_diff": a_flag_wr - b_flag_wr,
        "recent_5_diff": a_recent5 - b_recent5,
        "win_streak_diff": (a_streak - b_streak) / 10.0,
        "opp_style_wr": opp_style_wr,
        "style_a": str(style_a), "style_b": str(style_b),
        # For transfer score computation
        "a_recent_wr": a_recent_wr, "age_a": age_a, "age_b": age_b,
        "a_kata_div": a_kata_div, "b_kata_div": b_kata_div,
        "continent_a": str(continent_a),
        "same_continent": int(
            str(safe_mode(df_scope[df_scope["Nom"] == nom_a]["Continent"], "")) ==
            str(safe_mode(df_scope[df_scope["Nom"] == nom_b]["Continent"], ""))
        ),
    }


def _compute_transfer_score(base_feats: dict, a_kata_wr: float, tour_rank: float) -> float:
    """Compute transfer score using the pre-trained transfer model."""
    tf_model, tf_features = _load_transfer_model()
    if tf_model is None:
        return 0.0

    log_rank_ratio = np.log1p(base_feats["b_ranking"]) - np.log1p(base_feats["a_ranking"])

    tf_dict = {
        "Diff_Ranking": base_feats["b_ranking"] - base_feats["a_ranking"],
        "Log_Ranking_Ratio": log_rank_ratio,
        "Diff_Age": base_feats["age_diff"],
        "Diff_Winrate": base_feats["winrate_adv"],
        "Diff_Form": base_feats["momentum_diff"],
        "Diff_Kata_WR": a_kata_wr,
        "A_H2H_Rate": base_feats["h2h_wr_a"],
        "Exp_Diff": base_feats["exp_diff"],
        "Same_Nation": base_feats["same_nation"],
        "Same_Continent": base_feats["same_continent"],
        "Same_Style": base_feats["same_style"],
        "Tour_Rank": tour_rank,
        "Is_K1": base_feats["is_k1"],
        "Is_Male": base_feats["is_male"],
        "Home_Advantage": base_feats["is_home"],
        "Diff_Kata_Diversity": base_feats["kata_diversity_diff"],
        "Ranking_x_Tour": log_rank_ratio * tour_rank,
        "Ranking_x_IsMale": log_rank_ratio * base_feats["is_male"],
    }
    tf_X = pd.DataFrame([tf_dict])[tf_features]
    return float(tf_model.predict(tf_X)[0])


def _predict_for_katas(model, ag, nom_a, nom_b, n_tour, katas, base_feats):
    """Predict win probability for each kata using the LightGBM model."""
    ok_index = ag.athlete_oppkata_losses.set_index(["Nom", "Opp_Kata"]) if not ag.athlete_oppkata_losses.empty else None
    ak_index = ag.athlete_kata.set_index(["Nom", "Kata"]) if not ag.athlete_kata.empty else None
    kt_index = ag.kata_tour.set_index(["Kata", "N_Tour"]) if not ag.kata_tour.empty else None
    ke_map = ag.kata_effect.set_index("Kata")["Kata_Effect"].to_dict() if not ag.kata_effect.empty else {}
    seen_index = ag.opponent_seen_kata.set_index(["Nom", "Opp_Kata"]) if not ag.opponent_seen_kata.empty else None
    consistency_index = ag.athlete_kata_consistency.set_index(["Nom", "Kata"]) if not ag.athlete_kata_consistency.empty else None

    tour_rank = TOUR_ORDER.get(str(n_tour), 5) / 10.0

    results = []
    for kata in katas:
        kata = str(kata)

        # Kata-specific features for A
        if ak_index is not None and (nom_a, kata) in ak_index.index:
            row = ak_index.loc[(nom_a, kata)]
            a_kata_wr_raw = float(row["Kata_WinRate_Smoothed"])
            a_kata_note = float(row["Kata_Note_Mean"]) if pd.notna(row["Kata_Note_Mean"]) else ag.global_note_mean
            a_kata_n = int(row["Kata_Total"])
        else:
            a_kata_wr_raw = 0.5
            a_kata_note = ag.global_note_mean
            a_kata_n = 0

        a_kata_wr = a_kata_wr_raw - 0.5  # Centered

        # A kata consistency
        a_kata_consistency = 0.0
        if consistency_index is not None and (nom_a, kata) in consistency_index.index:
            std_val = float(consistency_index.loc[(nom_a, kata), "Kata_Note_Std"])
            a_kata_consistency = -std_val if std_val > 0 else 0.0

        # B weakness vs A's kata
        b_loss_vs = 0.5
        if ok_index is not None and (nom_b, kata) in ok_index.index:
            b_loss_vs = float(ok_index.loc[(nom_b, kata), "LossRate_vs_Kata_Smoothed"])

        # Kata × tour WR
        kata_tour_wr = 0.5
        if kt_index is not None and (kata, str(n_tour)) in kt_index.index:
            kata_tour_wr = float(kt_index.loc[(kata, str(n_tour)), "Kata_Tour_WinRate_Smoothed"])

        # B has seen this kata
        b_seen = 0
        if seen_index is not None and (nom_b, kata) in seen_index.index:
            b_seen = int(seen_index.loc[(nom_b, kata), "Seen_Count"] >= 2)

        # Favourite kata
        is_fav = int(base_feats["a_fav_kata"] == kata) if base_feats["a_fav_kata"] else 0

        note_diff = a_kata_note - base_feats["b_note_mean"]
        a_kata_note_c = a_kata_note - ag.global_note_mean
        log_a_kata_n = float(np.log1p(a_kata_n))

        # Transfer score
        transfer_score = _compute_transfer_score(base_feats, a_kata_wr, tour_rank)

        # Interactions
        ranking_x_tour = base_feats["ranking_adv"] * tour_rank
        winrate_x_tour = base_feats["winrate_adv"] * tour_rank
        katawr_x_tour = a_kata_wr * tour_rank

        # Build feature vector (ORDER MUST MATCH MODEL_FEATURES)
        feat_dict = {
            "WinRate_Adv": base_feats["winrate_adv"],
            "Ranking_Adv": base_feats["ranking_adv"],
            "Exp_Diff": base_feats["exp_diff"],
            "Momentum_Diff": base_feats["momentum_diff"],
            "Recent_5_Diff": base_feats["recent_5_diff"],
            "Note_Trend_Diff": base_feats["note_trend_diff"],
            "Note_Std_Diff": base_feats["note_std_diff"],
            "Age_Diff": base_feats["age_diff"],
            "Flag_WR_Diff": base_feats["flag_wr_diff"],
            "Win_Streak_Diff": base_feats["win_streak_diff"],
            "A_Kata_WR": a_kata_wr,
            "Log_A_Kata_N": log_a_kata_n,
            "A_Kata_Note_C": a_kata_note_c,
            "Note_Diff": note_diff,
            "A_Kata_Consistency": a_kata_consistency,
            "Is_Fav_Kata_A": is_fav,
            "H2H_Adv": base_feats["h2h_adv"],
            "B_Weakness_C": b_loss_vs - 0.5,
            "B_Seen_Kata": b_seen,
            "Opp_Style_WR": base_feats["opp_style_wr"],
            "Kata_Tour_C": kata_tour_wr - 0.5,
            "Kata_Effect": float(ke_map.get(kata, 0.0)),
            "Same_Nation": base_feats["same_nation"],
            "Same_Style": base_feats["same_style"],
            "Is_K1": base_feats["is_k1"],
            "Is_Male": base_feats["is_male"],
            "Is_Red": 1,  # A is red (aka)
            "Is_Home": base_feats["is_home"],
            "Tour_Rank": tour_rank,
            "Transfer_Score": transfer_score,
            "Ranking_x_Tour": ranking_x_tour,
            "WinRate_x_Tour": winrate_x_tour,
            "KataWR_x_Tour": katawr_x_tour,
            "Kata_Diversity_Diff": base_feats["kata_diversity_diff"],
        }

        X = pd.DataFrame([feat_dict])[MODEL_FEATURES]
        p_model = float(model.predict_proba(X)[0, 1])

        # Shrinkage (confidence = f(data available))
        w = _shrink_weight(int(base_feats["a_matches"]), int(base_feats["b_matches"]), int(a_kata_n))
        p_final = float(np.clip(0.5 + w * (p_model - 0.5), 0.01, 0.99))

        results.append({
            "Kata": kata,
            "Probabilité de victoire (%)": round(p_final * 100.0, 2),
            "Confiance (0-1)": round(w, 3),
            "Proba brute (%)": round(p_model * 100.0, 1),
            "Note moy. A (kata)": round(a_kata_note, 2),
            "Diff. notes (A-B)": round(note_diff, 2),
            "Nb occ. (A, kata)": int(a_kata_n),
            "Kata favori A": "✓" if is_fav else "",
            "B connaît kata": "✓" if b_seen else "",
        })

    return pd.DataFrame(results).sort_values("Probabilité de victoire (%)", ascending=False).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _shrink_weight(a_matches, b_matches, a_kata_n):
    """Confidence weight (higher = more data → less shrinkage toward 50%)."""
    w_a = 1.0 - np.exp(-a_matches / 10.0)
    w_b = 1.0 - np.exp(-b_matches / 10.0)
    w_k = 1.0 - np.exp(-a_kata_n / 8.0)
    w = 0.40 * ((w_a + w_b) / 2.0) + 0.60 * w_k
    return float(np.clip(w, 0.08, 0.92))


def _get_katas_of_style_in_scope(df_scope, style_a):
    katas_scope = df_scope["Kata"].dropna().astype(str).unique().tolist()
    if style_a is None or "Style" not in df_scope.columns:
        return sorted(set(map(str, katas_scope)))
    tmp = df_scope[["Kata", "Style"]].dropna()
    katas_style = sorted(tmp[tmp["Style"].astype(str) == str(style_a)]["Kata"].astype(str).unique().tolist())
    if "Suparinpei" in katas_scope and "Suparinpei" not in katas_style:
        katas_style.append("Suparinpei")
    return sorted(set(katas_style))


# ═══════════════════════════════════════════════════════════════════════════════
# Streamlit Tab — Main
# ═══════════════════════════════════════════════════════════════════════════════

def show_proba_victoire_kata_tab(data: pd.DataFrame) -> None:
    st.header(t("Probabilité de victoire par kata"))
    show_tab_help("proba_victoire")

    if get_lang() == "en":
        st.markdown(
            """
**Model v5 — LightGBM (pre-trained, AUC = 0.977)**
- Uses **34 features**: ranking, win rate, H2H, momentum (5 & 15 matches), note trend, transfer score, kata consistency, opponent weakness, flag performance, age, experience, win streak, kata diversity, interactions.
- Anti-bias: if A/B have little history, probability is **pulled toward 50%** (shrinkage).
- Results are **probabilities**, not certainties.
            """
        )
    else:
        st.markdown(
            """
**Modèle v5 — LightGBM (pré-entraîné, AUC = 0.977)**
- Utilise **34 features** : ranking, win rate, H2H, momentum (5 & 15 matchs), tendance de notes, transfer score, consistance kata, faiblesse adversaire, performance drapeaux, âge, expérience, série en cours, diversité kata, interactions.
- Anti-biais : si A/B ont peu d'historique, on **ramène la proba vers 50%** (shrinkage).
- Les résultats sont des **probabilités**, pas des certitudes.
            """
        )

    # Load pre-trained model
    model, model_features = _load_kata_victory_model()
    if model is None:
        st.error(t("Modèle introuvable. Lancez d'abord train.py dans model/proba_kata_victory/."))
        return

    df = data.copy()
    if "Note" in df.columns and df["Note"].dtype == object:
        df["Note"] = df["Note"].astype(str).str.replace(",", ".", regex=False)
    for col in ["Year", "Note", "Ranking", "Age"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Compute aggregates
    matches_scope, directed, ag = _get_aggregates(df)

    # ── Layout ──
    filters_col, content_col = st.columns([0.9, 2.4])

    run_manual = False
    run_top3 = False

    with filters_col:
        filter_panel_open()
        st.markdown(t("### 🎯 Paramètres"))

        type_compet_options = [t("Tous"), "Premier League (K1)", "Series A (SA)"]
        selected_type_compet = st.radio(t("Type de compétition"), type_compet_options, key="proba_type_compet")

        df_scope = df.copy()
        if selected_type_compet == "Premier League (K1)":
            df_scope = df_scope[df_scope["Type_Compet"] == "K1"]
        elif selected_type_compet == "Series A (SA)":
            df_scope = df_scope[df_scope["Type_Compet"] == "SA"]

        athlete_names = sorted(df_scope["Nom"].dropna().unique().tolist())
        if not athlete_names:
            st.warning(t("Aucun athlète disponible avec ces filtres."))
            filter_panel_close()
            return

        nom_a = st.selectbox(t("Athlète A (aka)"), athlete_names, key="proba_nom_a")

        sexe_a = safe_mode(df_scope[df_scope["Nom"] == nom_a]["Sexe"], default=None)
        if sexe_a is not None:
            athlete_b_names = sorted(
                df_scope[(df_scope["Sexe"] == sexe_a) & (df_scope["Nom"] != nom_a)]["Nom"].dropna().unique().tolist()
            )
        else:
            athlete_b_names = sorted(df_scope[df_scope["Nom"] != nom_a]["Nom"].dropna().unique().tolist())

        if not athlete_b_names:
            st.warning(t("Aucun adversaire compatible trouvé."))
            filter_panel_close()
            return

        nom_b = st.selectbox(t("Athlète B (shiro)"), athlete_b_names, key="proba_nom_b")

        tour_options = sorted(df_scope["N_Tour"].dropna().astype(str).unique().tolist())
        if not tour_options:
            st.warning(t("Aucun tour disponible."))
            filter_panel_close()
            return

        n_tour = st.selectbox(t("Tour simulé"), tour_options, format_func=fmt_tour, key="proba_tour")

        style_a = safe_mode(df_scope[df_scope["Nom"] == nom_a]["Style"], default=None)

        st.markdown("---")
        st.markdown(t("#### 🥋 Katas testés (pour A)"))

        katas_style = _get_katas_of_style_in_scope(df_scope, style_a)
        katas_effectues_a = sorted(df_scope[df_scope["Nom"] == nom_a]["Kata"].dropna().astype(str).unique().tolist())
        katas_effectues_a = [k for k in katas_effectues_a if k in katas_style]

        katas_selectionnes = st.multiselect(
            t("Katas à tester"), options=katas_style,
            default=katas_effectues_a if katas_effectues_a else katas_style,
            key="proba_katas",
        )

        st.markdown("---")
        st.markdown(t("#### 🏆 Top 3 katas à faire"))
        if get_lang() == "en":
            st.caption(
                "Based on opponent and round.\n\n"
                "- If A has **≥ 4 matches** ⇒ Top 3 from their **already played katas**.\n"
                "- Otherwise ⇒ Top 3 from **all style katas**."
            )
        else:
            st.caption(
                "Basé sur l'adversaire et le tour.\n\n"
                "- Si A a **≥ 4 matchs** ⇒ Top 3 parmi ses **katas déjà joués**.\n"
                "- Sinon ⇒ Top 3 parmi **tous les katas du style**."
            )
        run_top3 = st.button(t("🏆 Top 3"), key="proba_top3")

        st.markdown("---")
        run_manual = st.button(t("🎯 Calculer les probabilités"), key="proba_run", type="primary")
        filter_panel_close()

    with content_col:
        # Quick stats
        c1, c2 = st.columns(2)
        with c1:
            a_stats_row = ag.athlete_stats[ag.athlete_stats["Nom"] == nom_a]
            if not a_stats_row.empty:
                r = a_stats_row.iloc[0]
                st.metric(f"A – {nom_a}", f"{r['WinRate_Smoothed']:.0%} WR",
                          help=t("Win rate global de A"))
                st.caption(f"{int(r['Wins'])}/{int(r['Total'])} {t('matchs')}")
                if pd.notna(r["Ranking_Mean"]):
                    st.caption(f"Ranking moy. : {r['Ranking_Mean']:.0f}")
            else:
                st.metric(f"A – {nom_a}", "?")
        with c2:
            b_stats_row = ag.athlete_stats[ag.athlete_stats["Nom"] == nom_b]
            if not b_stats_row.empty:
                r = b_stats_row.iloc[0]
                st.metric(f"B – {nom_b}", f"{r['WinRate_Smoothed']:.0%} WR",
                          help=t("Win rate global de B"))
                st.caption(f"{int(r['Wins'])}/{int(r['Total'])} {t('matchs')}")
                if pd.notna(r["Ranking_Mean"]):
                    st.caption(f"Ranking moy. : {r['Ranking_Mean']:.0f}")
            else:
                st.metric(f"B – {nom_b}", "?")

        # Model info
        with st.expander(t("📊 Performance du modèle")):
            st.markdown(
                f"- **Modèle** : LightGBM + Optuna (pré-entraîné)\n"
                f"- **Accuracy (test)** : 92.7%\n"
                f"- **AUC ROC (test)** : 0.977\n"
                f"- **AUC CV (5-fold)** : 0.944\n"
                f"- **Features** : 34\n"
                f"- **Train/Test split** : chronologique (80/20)"
            )

            st.markdown(t("##### Top features (importance)"))
            st.caption("Transfer_Score, Ranking_Adv, H2H_Adv, Note_Diff, WinRate_Adv, "
                       "Momentum_Diff, A_Kata_WR, Exp_Diff, Kata_Diversity_Diff, Note_Trend_Diff")

        # ── Predictions ──
        base_feats = _compute_base_features(df_scope, ag, matches_scope, selected_type_compet, nom_a, nom_b)

        if run_manual and katas_selectionnes:
            res_df = _predict_for_katas(
                model=model, ag=ag, nom_a=nom_a, nom_b=nom_b,
                n_tour=str(n_tour), katas=list(map(str, katas_selectionnes)),
                base_feats=base_feats,
            )

            st.subheader(f"{t('Résultats')} – {nom_a} vs {nom_b} ({t('tour')}: {fmt_tour(n_tour)})")

            # Top 3 barres colorées
            st.markdown(f"##### Top 3 ({t('sur ta sélection')})")
            for _, row in res_df.head(3).iterrows():
                p = row["Probabilité de victoire (%)"]
                col_kata, col_bar, col_conf = st.columns([1, 2, 1])
                with col_kata:
                    st.markdown(f"**{row['Kata']}**")
                with col_bar:
                    st.markdown(proba_bar_html(p), unsafe_allow_html=True)
                with col_conf:
                    conf = row["Confiance (0-1)"]
                    conf_color = "green" if conf >= 0.6 else "orange" if conf >= 0.3 else "red"
                    st.markdown(f"{t('Confiance')}: {_color_badge(f'{conf:.2f}', conf_color)}", unsafe_allow_html=True)

            st.markdown(f"##### {t('Détail complet')}")
            st.dataframe(format_display_df(res_df), use_container_width=True)

            fig = px.bar(
                res_df, x="Kata", y="Probabilité de victoire (%)",
                color="Probabilité de victoire (%)",
                color_continuous_scale=["#dc3545", "#ffc107", "#28a745"],
                range_color=[30, 70],
                hover_data=["Confiance (0-1)", "Proba brute (%)", "Diff. notes (A-B)", "Nb occ. (A, kata)"],
                title=t("Probabilité de victoire par kata (A) – sélection"),
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True, key="proba_kata_bar_manual")

        if run_top3:
            a_match_count = int(base_feats["a_matches"])
            if a_match_count >= 4:
                candidates = sorted(set(
                    df_scope[df_scope["Nom"] == nom_a]["Kata"].dropna().astype(str).unique().tolist()
                ))
                source = t("katas déjà joués par A") + f" (A: {a_match_count} {t('matchs')})"
            else:
                candidates = list(map(str, katas_style))
                source = t("tous les katas du style") + f" (A: {a_match_count} {t('matchs')})"

            if not candidates:
                st.warning(t("Impossible de proposer un Top 3 : aucun kata candidat trouvé."))
            else:
                res_df_top = _predict_for_katas(
                    model=model, ag=ag, nom_a=nom_a, nom_b=nom_b,
                    n_tour=str(n_tour), katas=candidates, base_feats=base_feats,
                )

                st.subheader(t("🏆 Top 3 katas à faire"))
                st.caption(f"{t('Source candidats')} : **{source}**")

                for _, row in res_df_top.head(3).iterrows():
                    p = row["Probabilité de victoire (%)"]
                    col_kata, col_bar, col_conf = st.columns([1, 2, 1])
                    with col_kata:
                        st.markdown(f"**{row['Kata']}**")
                    with col_bar:
                        st.markdown(proba_bar_html(p), unsafe_allow_html=True)
                    with col_conf:
                        conf = row["Confiance (0-1)"]
                        conf_color = "green" if conf >= 0.6 else "orange" if conf >= 0.3 else "red"
                        st.markdown(f"{t('Confiance')}: {_color_badge(f'{conf:.2f}', conf_color)}", unsafe_allow_html=True)

                st.dataframe(format_display_df(res_df_top), use_container_width=True)

                fig2 = px.bar(
                    res_df_top.head(10), x="Kata", y="Probabilité de victoire (%)",
                    color="Probabilité de victoire (%)",
                    color_continuous_scale=["#dc3545", "#ffc107", "#28a745"],
                    range_color=[30, 70],
                    hover_data=["Confiance (0-1)", "Proba brute (%)", "Nb occ. (A, kata)"],
                    title=t("Top katas recommandés (Top 10)"),
                )
                fig2.update_layout(height=400)
                st.plotly_chart(fig2, use_container_width=True, key="proba_kata_bar_top3")

        if not run_manual and not run_top3:
            st.info(t("Sélectionnez les paramètres puis cliquez sur 'Calculer les probabilités' ou 'Top 3'."))
