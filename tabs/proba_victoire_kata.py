# tabs/proba_victoire_kata.py  — v2 : modèle corrigé
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils.ui import filter_panel_open, filter_panel_close
from utils.data_helpers import victoire_to_int, safe_mode, compute_time_weight, is_flag_era
from utils.interpretations import show_tab_help, proba_bar_html, proba_color, _color_badge
from utils.display import fmt_tour, format_display_df
from utils.lang import t, get_lang


# ═══════════════════════════════════════════════════════════════════════════════
# Small helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _to_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def _beta_smooth_rate(wins: float, total: float, alpha: float = 2.0, beta: float = 2.0) -> float:
    """Bayesian-smoothed win rate.  Higher alpha/beta → more shrinkage to 0.5."""
    return (wins + alpha) / (total + alpha + beta)


def _center_rate(x: float) -> float:
    try:
        return float(x) - 0.5
    except Exception:
        return 0.0


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


# ═══════════════════════════════════════════════════════════════════════════════
# Pairing R/B → matches (with sort for robustness)
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _build_paired_matches(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy().reset_index(drop=True)

    needed = ["Nom", "Ceinture", "Kata", "N_Tour", "Competition", "Type_Compet", "Victoire"]
    for c in needed:
        if c not in d.columns:
            raise ValueError(f"Colonne manquante: {c}")

    for col_default in ["Year", "Nation", "Style", "Sexe", "Note", "Ranking", "Age", "Continent", "Region_monde", "Drapeau"]:
        if col_default not in d.columns:
            d[col_default] = np.nan

    # Filter out placeholder rows (no Nom)
    d = d[d["Nom"].notna() & (d["Nom"].astype(str).str.strip() != "")].reset_index(drop=True)

    d = d.sort_values(
        ["Competition", "Year", "Type_Compet", "N_Tour"], kind="mergesort",
    ).reset_index(drop=True)

    # ── Vectorized pairing via shift ──
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

    # Victoire of the red athlete
    v1 = r1["Victoire"].astype(str).str.lower().isin(["true", "1", "vrai", "yes"]).astype(int).values
    v2 = r2["Victoire"].astype(str).str.lower().isin(["true", "1", "vrai", "yes"]).astype(int).values
    red_win = np.where(is_r1_red, v1, v2)

    # Note conversion
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

    # Red perspective (vectorized)
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
    red["Opp_Sexe"] = matches["Blue_Sexe"]
    red["Note"] = matches["Red_Note"]
    red["Opp_Note"] = matches["Blue_Note"]
    red["Ranking"] = matches["Red_Ranking"]
    red["Opp_Ranking"] = matches["Blue_Ranking"]
    red["Age"] = matches["Red_Age"]
    red["Opp_Age"] = matches["Blue_Age"]
    red["Continent"] = matches["Red_Continent"]
    red["Opp_Continent"] = matches["Blue_Continent"]
    red["Region"] = matches["Red_Region"]
    red["Opp_Region"] = matches["Blue_Region"]
    red["Athlete_Win"] = matches["Red_Win"].astype(int)
    red["Is_Red"] = 1

    # Blue perspective (vectorized)
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
    blue["Opp_Sexe"] = matches["Red_Sexe"]
    blue["Note"] = matches["Blue_Note"]
    blue["Opp_Note"] = matches["Red_Note"]
    blue["Ranking"] = matches["Blue_Ranking"]
    blue["Opp_Ranking"] = matches["Red_Ranking"]
    blue["Age"] = matches["Blue_Age"]
    blue["Opp_Age"] = matches["Red_Age"]
    blue["Continent"] = matches["Blue_Continent"]
    blue["Opp_Continent"] = matches["Red_Continent"]
    blue["Region"] = matches["Blue_Region"]
    blue["Opp_Region"] = matches["Red_Region"]
    blue["Athlete_Win"] = (1 - matches["Red_Win"]).astype(int)
    blue["Is_Red"] = 0

    return pd.concat([red, blue], ignore_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Aggregates
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
    # v3 — new aggregate tables
    athlete_trend: pd.DataFrame       # Note_Trend, Note_Std per athlete
    athlete_recent: pd.DataFrame      # Recent_WinRate per athlete (last 15)
    athlete_fav_kata: pd.DataFrame    # favourite kata per athlete
    athlete_kata_diversity: pd.DataFrame  # Kata_Diversity per athlete
    opponent_seen_kata: pd.DataFrame  # B has seen kata before (Nom, Opp_Kata, Seen)
    # v4 — flag era & short-term dynamics
    athlete_flag_stats: pd.DataFrame  # Flag-era WR per athlete
    athlete_recent_5: pd.DataFrame    # Short momentum (last 5 matches)
    athlete_streak: pd.DataFrame      # Current win/loss streak


def _compute_aggregates(directed: pd.DataFrame) -> Aggregates:
    if directed.empty:
        empty = pd.DataFrame()
        return Aggregates(empty, empty, empty, empty, empty, empty, 0.0,
                          empty, empty, empty, empty, empty,
                          empty, empty, empty)

    d = directed.copy()

    # Global note mean (for centering)
    global_note_mean = float(d["Note"].dropna().mean()) if d["Note"].notna().any() else 0.0

    # --- Athlete-level stats ---
    a = d.groupby("Nom").agg(
        Wins=("Athlete_Win", "sum"),
        Total=("Athlete_Win", "count"),
        Note_Mean=("Note", "mean"),
        Ranking_Mean=("Ranking", "mean"),
    ).reset_index()
    a["WinRate_Smoothed"] = (a["Wins"] + 2) / (a["Total"] + 4)

    # --- Athlete × Kata stats (beta prior 3.0 instead of 1.5) ---
    ak = d.groupby(["Nom", "Kata"])["Athlete_Win"].agg(["sum", "count"]).reset_index()
    ak.rename(columns={"sum": "Kata_Wins", "count": "Kata_Total"}, inplace=True)
    notes = d.dropna(subset=["Note"]).groupby(["Nom", "Kata"])["Note"].mean().reset_index()
    notes.rename(columns={"Note": "Kata_Note_Mean"}, inplace=True)
    ak = ak.merge(notes, on=["Nom", "Kata"], how="left")
    # ✅ FIX: stronger prior (3.0 instead of 1.5) to reduce small-sample bias
    ak["Kata_WinRate_Smoothed"] = (ak["Kata_Wins"] + 3.0) / (ak["Kata_Total"] + 6.0)

    # --- Opponent weakness vs kata (also 3.0) ---
    d["Loss"] = 1 - d["Athlete_Win"]
    ok = d.groupby(["Nom", "Opp_Kata"])["Loss"].agg(["sum", "count"]).reset_index()
    ok.rename(columns={"sum": "Losses_vs_Kata", "count": "Total_vs_Kata"}, inplace=True)
    ok["LossRate_vs_Kata_Smoothed"] = (ok["Losses_vs_Kata"] + 3.0) / (ok["Total_vs_Kata"] + 6.0)

    # --- Kata × Tour ---
    kt = d.groupby(["Kata", "N_Tour"])["Athlete_Win"].agg(["sum", "count"]).reset_index()
    kt.rename(columns={"sum": "Wins", "count": "Total"}, inplace=True)
    kt["Kata_Tour_WinRate_Smoothed"] = (kt["Wins"] + 2) / (kt["Total"] + 4)

    # --- Head-to-head ---
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

    # --- Kata residual effect ---
    a_map = a.set_index("Nom")["WinRate_Smoothed"].to_dict()
    d["Athlete_Base"] = d["Nom"].map(a_map).fillna(0.5)
    d["Residual_vs_Base"] = d["Athlete_Win"] - d["Athlete_Base"]
    ke = d.groupby("Kata")["Residual_vs_Base"].agg(["mean", "count"]).reset_index()
    ke.rename(columns={"mean": "Residual_Mean", "count": "Kata_Uses"}, inplace=True)
    k = 25.0
    ke["Kata_Effect"] = ke["Residual_Mean"] * (ke["Kata_Uses"] / (ke["Kata_Uses"] + k))

    # ── v3 NEW aggregates (all vectorized) ──

    # 1) Note trend (slope) + Note_Std per athlete
    #    We assign a match order per athlete and do a grouped linear regression via cov/var
    d_notes = d.dropna(subset=["Note"]).copy()
    d_notes["_match_idx"] = d_notes.groupby("Nom").cumcount()
    g_trend = d_notes.groupby("Nom").agg(
        _n=("_match_idx", "count"),
        _mean_x=("_match_idx", "mean"),
        _mean_y=("Note", "mean"),
        _var_x=("_match_idx", "var"),
        Note_Std=("Note", "std"),
    ).reset_index()
    # covariance via E[xy] - E[x]*E[y]
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

    # 2) Recent win rate (last 15 matches per athlete)
    d_recent = d.copy()
    d_recent["_rev_idx"] = d_recent.groupby("Nom").cumcount(ascending=False)
    recent_15 = d_recent[d_recent["_rev_idx"] < 15]
    r_wr = recent_15.groupby("Nom")["Athlete_Win"].agg(["sum", "count"]).reset_index()
    r_wr.rename(columns={"sum": "Recent_Wins", "count": "Recent_Total"}, inplace=True)
    r_wr["Recent_WinRate"] = (r_wr["Recent_Wins"] + 2) / (r_wr["Recent_Total"] + 4)
    athlete_recent = r_wr[["Nom", "Recent_WinRate"]].copy()

    # 3) Favourite kata per athlete (most used kata)
    fav = ak.sort_values("Kata_Total", ascending=False).drop_duplicates("Nom", keep="first")
    athlete_fav_kata = fav[["Nom", "Kata"]].rename(columns={"Kata": "Fav_Kata"}).copy()

    # 4) Kata diversity per athlete (nb distinct katas / total matches)
    kata_div = d.groupby("Nom").agg(
        Nb_Katas=("Kata", "nunique"),
        Nb_Matchs=("Athlete_Win", "count"),
    ).reset_index()
    kata_div["Kata_Diversity"] = kata_div["Nb_Katas"] / kata_div["Nb_Matchs"]
    athlete_kata_diversity = kata_div[["Nom", "Kata_Diversity"]].copy()

    # 5) Opponent seen kata (has B faced this Opp_Kata before)
    opp_seen = d.groupby(["Nom", "Opp_Kata"])["Athlete_Win"].count().reset_index()
    opp_seen.rename(columns={"Athlete_Win": "Seen_Count"}, inplace=True)
    opp_seen["Seen"] = (opp_seen["Seen_Count"] > 0).astype(int)

    # ── v4 NEW aggregates: flag era, short-term momentum, streak ──

    # 6) Flag-era win rate per athlete (matches where Note is NaN → flag era)
    d_flag = d[d["Note"].isna()].copy()
    if not d_flag.empty:
        flag_agg = d_flag.groupby("Nom")["Athlete_Win"].agg(["sum", "count"]).reset_index()
        flag_agg.rename(columns={"sum": "Flag_Wins", "count": "Flag_Total"}, inplace=True)
        flag_agg["Flag_WinRate"] = (flag_agg["Flag_Wins"] + 2) / (flag_agg["Flag_Total"] + 4)
        athlete_flag_stats = flag_agg[["Nom", "Flag_WinRate", "Flag_Total"]].copy()
    else:
        athlete_flag_stats = pd.DataFrame(columns=["Nom", "Flag_WinRate", "Flag_Total"])

    # 7) Very recent momentum (last 5 matches) — more responsive than last 15
    d_recent5 = d.copy()
    d_recent5["_rev_idx"] = d_recent5.groupby("Nom").cumcount(ascending=False)
    recent_5 = d_recent5[d_recent5["_rev_idx"] < 5]
    r5_wr = recent_5.groupby("Nom")["Athlete_Win"].agg(["sum", "count"]).reset_index()
    r5_wr.rename(columns={"sum": "Recent5_Wins", "count": "Recent5_Total"}, inplace=True)
    r5_wr["Recent5_WinRate"] = (r5_wr["Recent5_Wins"] + 1) / (r5_wr["Recent5_Total"] + 2)
    athlete_recent_5 = r5_wr[["Nom", "Recent5_WinRate"]].copy()

    # 8) Current win/loss streak (positive = win streak, negative = loss streak)
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

    return Aggregates(a, ak, ok, kt, h2h, ke, global_note_mean,
                      athlete_trend, athlete_recent, athlete_fav_kata,
                      athlete_kata_diversity, opp_seen,
                      athlete_flag_stats, athlete_recent_5, athlete_streak)


# ═══════════════════════════════════════════════════════════════════════════════
# Model: features & training — v2
# ═══════════════════════════════════════════════════════════════════════════════

FEATURE_COLS = [
    "Same_Nation", "Same_Style", "Is_K1",
    "H2H_Adv", "WinRate_Adv",
    "A_Kata_WinRate_C", "B_Weakness_C",
    "Kata_Tour_C", "Kata_Effect",
    "Note_Diff", "Ranking_Adv", "Log_A_Kata_N", "A_Kata_Note_C",
    # v3 — temporal & context features
    "Note_Trend_Diff",   # slope of A's notes - slope of B's notes
    "Momentum_Diff",     # recent win rate A - recent win rate B (last 15)
    "Age_Diff",          # (Age_A - Age_B) / 10
    "Is_Fav_Kata_A",     # A plays their most frequent kata (0/1)
    "Note_Std_Diff",     # consistency: std A - std B (negative = A more consistent)
    "Exp_Diff",          # log(matchs A) - log(matchs B)
    "Kata_Diversity_Diff",  # diversity A - diversity B
    "B_Seen_Kata",       # B has faced this kata before (0/1)
    "Is_Home",           # A plays in their continent
    "Tour_Rank",         # ordinal encoding of round (higher = later)
    # v4 — flag, ceinture & short-term dynamics
    "Is_Red",            # A wears the red belt (0/1)
    "Flag_WinRate_Diff", # A's flag win rate - B's flag win rate (2026+)
    "Recent_5_Diff",     # A's last 5 matches WR - B's (hot hand)
    "Win_Streak_Diff",   # A's streak - B's streak (normalized)
]

# Tour ordinal ranking for Tour_Rank feature
_TOUR_RANK_MAP = {
    "pool_1": 1, "pool_2": 2, "pool_3": 3, "pool_4": 4,
    "round_1": 2, "round_2": 3, "1/8": 3, "1/4": 4,
    "rp1": 3, "rp2": 3, "rp3": 4, "rp4": 4,
    "bronze": 5, "finale": 6, "final": 6,
}


def _tour_to_rank(tour_str: str) -> float:
    return _TOUR_RANK_MAP.get(str(tour_str).strip().lower(), 2.0)


def _prepare_training_table(directed: pd.DataFrame, ag: Aggregates) -> pd.DataFrame:
    if directed.empty:
        return pd.DataFrame()

    d = directed.copy()
    d["Same_Nation"] = (d["Nation"].astype(str) == d["Opp_Nation"].astype(str)).astype(int)
    d["Same_Style"] = (d["Style"].astype(str) == d["Opp_Style"].astype(str)).astype(int)
    d["Is_K1"] = (d["Type_Compet"].astype(str) == "K1").astype(int)

    # Win rate advantage (vectorized map)
    a_map = ag.athlete_stats.set_index("Nom")["WinRate_Smoothed"]
    d["A_WinRate"] = d["Nom"].map(a_map).fillna(0.5)
    d["B_WinRate"] = d["Opponent"].map(a_map).fillna(0.5)
    d["WinRate_Adv"] = d["A_WinRate"] - d["B_WinRate"]

    # A kata stats — merge instead of apply+index lookup
    ak_cols = ag.athlete_kata[["Nom", "Kata", "Kata_WinRate_Smoothed", "Kata_Note_Mean", "Kata_Total"]].copy()
    d = d.merge(ak_cols, on=["Nom", "Kata"], how="left")
    d["Kata_WinRate_Smoothed"] = d["Kata_WinRate_Smoothed"].fillna(0.5)
    d["A_Kata_WinRate_C"] = d["Kata_WinRate_Smoothed"] - 0.5
    d["Kata_Note_Mean"] = d["Kata_Note_Mean"].fillna(ag.global_note_mean)
    d["A_Kata_Note_C"] = d["Kata_Note_Mean"] - ag.global_note_mean
    d["Kata_Total"] = d["Kata_Total"].fillna(0)
    d["Log_A_Kata_N"] = np.log1p(d["Kata_Total"])

    # Note diff — merge B's average note
    b_note = ag.athlete_stats[["Nom", "Note_Mean"]].rename(columns={"Nom": "Opponent", "Note_Mean": "B_Note_Mean"})
    d = d.merge(b_note, on="Opponent", how="left")
    d["B_Note_Mean"] = d["B_Note_Mean"].fillna(ag.global_note_mean)
    d["Note_Diff"] = d["Kata_Note_Mean"] - d["B_Note_Mean"]

    # Ranking advantage
    d["Ranking"] = pd.to_numeric(d["Ranking"], errors="coerce")
    d["Opp_Ranking"] = pd.to_numeric(d["Opp_Ranking"], errors="coerce")
    median_rank = d["Ranking"].median()
    if pd.isna(median_rank):
        median_rank = 100.0
    d["Ranking"] = d["Ranking"].fillna(median_rank)
    d["Opp_Ranking"] = d["Opp_Ranking"].fillna(median_rank)
    d["Ranking_Adv"] = (d["Opp_Ranking"] - d["Ranking"]) / 100.0

    # B weakness vs kata — merge instead of apply
    ok_cols = ag.athlete_oppkata_losses[["Nom", "Opp_Kata", "LossRate_vs_Kata_Smoothed"]].rename(
        columns={"Nom": "Opponent", "Opp_Kata": "Kata", "LossRate_vs_Kata_Smoothed": "B_LossRate"}
    )
    d = d.merge(ok_cols, on=["Opponent", "Kata"], how="left")
    d["B_LossRate"] = d["B_LossRate"].fillna(0.5)
    d["B_Weakness_C"] = d["B_LossRate"] - 0.5

    # Kata × tour — merge instead of apply
    kt_cols = ag.kata_tour[["Kata", "N_Tour", "Kata_Tour_WinRate_Smoothed"]].copy()
    kt_cols["N_Tour"] = kt_cols["N_Tour"].astype(str)
    d["N_Tour"] = d["N_Tour"].astype(str)
    d = d.merge(kt_cols, on=["Kata", "N_Tour"], how="left")
    d["Kata_Tour_WinRate_Smoothed"] = d["Kata_Tour_WinRate_Smoothed"].fillna(0.5)
    d["Kata_Tour_C"] = d["Kata_Tour_WinRate_Smoothed"] - 0.5

    # Kata effect (map is already vectorized)
    ke = ag.kata_effect.set_index("Kata")["Kata_Effect"]
    d["Kata_Effect"] = d["Kata"].map(ke).fillna(0.0)

    # Head-to-head — vectorized merge instead of apply
    nom_s = d["Nom"].astype(str)
    opp_s = d["Opponent"].astype(str)
    d["_h2h_A"] = np.where(nom_s <= opp_s, nom_s, opp_s)
    d["_h2h_B"] = np.where(nom_s > opp_s, nom_s, opp_s)

    if not ag.h2h.empty:
        h_cols = ag.h2h[["A", "B", "Total", "WinRate_A_Smoothed"]].rename(
            columns={"A": "_h2h_A", "B": "_h2h_B", "Total": "H2H_Total", "WinRate_A_Smoothed": "H2H_WR_A"}
        )
        d = d.merge(h_cols, on=["_h2h_A", "_h2h_B"], how="left")
        d["H2H_Total"] = d["H2H_Total"].fillna(0)
        d["H2H_WR_A"] = d["H2H_WR_A"].fillna(0.5)
        is_nom_A = nom_s == d["_h2h_A"]
        d["H2H_WR"] = np.where(is_nom_A, d["H2H_WR_A"], 1.0 - d["H2H_WR_A"])
    else:
        d["H2H_Total"] = 0.0
        d["H2H_WR"] = 0.5

    d["H2H_Adv"] = (d["H2H_WR"] - 0.5) * np.log1p(d["H2H_Total"])

    # ── v3 NEW features (all vectorized merges) ──

    # Note_Trend_Diff — slope of A's note progression minus B's
    if not ag.athlete_trend.empty:
        trend_a = ag.athlete_trend[["Nom", "Note_Trend", "Note_Std"]].copy()
        d = d.merge(trend_a, on="Nom", how="left", suffixes=("", "_A_trend"))
        trend_b = ag.athlete_trend[["Nom", "Note_Trend", "Note_Std"]].rename(
            columns={"Nom": "Opponent", "Note_Trend": "B_Note_Trend", "Note_Std": "B_Note_Std"}
        )
        d = d.merge(trend_b, on="Opponent", how="left")
        d["Note_Trend"] = d["Note_Trend"].fillna(0.0)
        d["B_Note_Trend"] = d["B_Note_Trend"].fillna(0.0)
        d["Note_Std"] = d["Note_Std"].fillna(0.0)
        d["B_Note_Std"] = d["B_Note_Std"].fillna(0.0)
    else:
        d["Note_Trend"] = 0.0
        d["B_Note_Trend"] = 0.0
        d["Note_Std"] = 0.0
        d["B_Note_Std"] = 0.0
    d["Note_Trend_Diff"] = d["Note_Trend"] - d["B_Note_Trend"]
    d["Note_Std_Diff"] = d["Note_Std"] - d["B_Note_Std"]

    # Momentum_Diff — recent win rate
    if not ag.athlete_recent.empty:
        rec_a = ag.athlete_recent.copy()
        d = d.merge(rec_a, on="Nom", how="left")
        rec_b = ag.athlete_recent.rename(columns={"Nom": "Opponent", "Recent_WinRate": "B_Recent_WinRate"})
        d = d.merge(rec_b, on="Opponent", how="left")
        d["Recent_WinRate"] = d["Recent_WinRate"].fillna(0.5)
        d["B_Recent_WinRate"] = d["B_Recent_WinRate"].fillna(0.5)
    else:
        d["Recent_WinRate"] = 0.5
        d["B_Recent_WinRate"] = 0.5
    d["Momentum_Diff"] = d["Recent_WinRate"] - d["B_Recent_WinRate"]

    # Age_Diff
    d["Age"] = pd.to_numeric(d["Age"], errors="coerce")
    d["Opp_Age"] = pd.to_numeric(d["Opp_Age"], errors="coerce")
    median_age = d["Age"].median()
    if pd.isna(median_age):
        median_age = 25.0
    d["Age"] = d["Age"].fillna(median_age)
    d["Opp_Age"] = d["Opp_Age"].fillna(median_age)
    d["Age_Diff"] = (d["Age"] - d["Opp_Age"]) / 10.0

    # Is_Fav_Kata_A — A plays their most frequent kata
    if not ag.athlete_fav_kata.empty:
        d = d.merge(ag.athlete_fav_kata, on="Nom", how="left")
        d["Is_Fav_Kata_A"] = (d["Kata"].astype(str) == d["Fav_Kata"].astype(str)).astype(int)
    else:
        d["Is_Fav_Kata_A"] = 0

    # Exp_Diff — log(total matches A) - log(total matches B)
    a_total = ag.athlete_stats[["Nom", "Total"]].rename(columns={"Total": "A_Total_Matchs"})
    b_total = ag.athlete_stats[["Nom", "Total"]].rename(columns={"Nom": "Opponent", "Total": "B_Total_Matchs"})
    d = d.merge(a_total, on="Nom", how="left")
    d = d.merge(b_total, on="Opponent", how="left")
    d["A_Total_Matchs"] = d["A_Total_Matchs"].fillna(1)
    d["B_Total_Matchs"] = d["B_Total_Matchs"].fillna(1)
    d["Exp_Diff"] = np.log1p(d["A_Total_Matchs"]) - np.log1p(d["B_Total_Matchs"])

    # Kata_Diversity_Diff
    if not ag.athlete_kata_diversity.empty:
        div_a = ag.athlete_kata_diversity.copy()
        d = d.merge(div_a, on="Nom", how="left")
        div_b = ag.athlete_kata_diversity.rename(columns={"Nom": "Opponent", "Kata_Diversity": "B_Kata_Diversity"})
        d = d.merge(div_b, on="Opponent", how="left")
        d["Kata_Diversity"] = d["Kata_Diversity"].fillna(0.5)
        d["B_Kata_Diversity"] = d["B_Kata_Diversity"].fillna(0.5)
    else:
        d["Kata_Diversity"] = 0.5
        d["B_Kata_Diversity"] = 0.5
    d["Kata_Diversity_Diff"] = d["Kata_Diversity"] - d["B_Kata_Diversity"]

    # B_Seen_Kata — has B faced this kata from an opponent before
    # In training data, current match counts as 1, so threshold >= 2 means real prior experience
    if not ag.opponent_seen_kata.empty:
        seen_cols = ag.opponent_seen_kata[["Nom", "Opp_Kata", "Seen_Count"]].rename(
            columns={"Nom": "Opponent", "Opp_Kata": "Kata", "Seen_Count": "B_Seen_Count"}
        )
        d = d.merge(seen_cols, on=["Opponent", "Kata"], how="left")
        d["B_Seen_Count"] = d["B_Seen_Count"].fillna(0)
        d["B_Seen_Kata"] = (d["B_Seen_Count"] >= 2).astype(int)
    else:
        d["B_Seen_Kata"] = 0

    # Is_Home — A plays in their continent
    d["Is_Home"] = (d["Continent"].astype(str) == d["Region"].astype(str)).astype(int)
    # Fallback: if Region not informative, try Continent match with competition region
    # (Region in directed comes from match Red/Blue_Region = Region_monde)

    # Tour_Rank — ordinal encoding of round
    d["Tour_Rank"] = d["N_Tour"].astype(str).map(_tour_to_rank).fillna(2.0)
    d["Tour_Rank"] = d["Tour_Rank"] / 6.0  # normalize to [0, 1]

    # v4 — Is_Red: already set in _to_directed_rows
    if "Is_Red" not in d.columns:
        d["Is_Red"] = 0

    # v4 — Flag_WinRate_Diff: actual flag-era win rate per athlete
    if not ag.athlete_flag_stats.empty:
        flag_a = ag.athlete_flag_stats[["Nom", "Flag_WinRate"]].copy()
        flag_b = ag.athlete_flag_stats[["Nom", "Flag_WinRate"]].rename(
            columns={"Nom": "Opponent", "Flag_WinRate": "B_Flag_WinRate"}
        )
        d = d.merge(flag_a, on="Nom", how="left")
        d = d.merge(flag_b, on="Opponent", how="left")
        d["Flag_WinRate"] = d["Flag_WinRate"].fillna(0.5)
        d["B_Flag_WinRate"] = d["B_Flag_WinRate"].fillna(0.5)
        d["Flag_WinRate_Diff"] = d["Flag_WinRate"] - d["B_Flag_WinRate"]
    else:
        d["Flag_WinRate_Diff"] = 0.0

    # v4 — Recent_5_Diff: very short-term momentum (last 5 matches)
    if not ag.athlete_recent_5.empty:
        r5_a = ag.athlete_recent_5.copy()
        r5_b = ag.athlete_recent_5.rename(columns={"Nom": "Opponent", "Recent5_WinRate": "B_Recent5_WinRate"})
        d = d.merge(r5_a, on="Nom", how="left")
        d = d.merge(r5_b, on="Opponent", how="left")
        d["Recent5_WinRate"] = d["Recent5_WinRate"].fillna(0.5)
        d["B_Recent5_WinRate"] = d["B_Recent5_WinRate"].fillna(0.5)
    else:
        d["Recent5_WinRate"] = 0.5
        d["B_Recent5_WinRate"] = 0.5
    d["Recent_5_Diff"] = d["Recent5_WinRate"] - d["B_Recent5_WinRate"]

    # v4 — Win_Streak_Diff: current win/loss streak (normalized by /10)
    if not ag.athlete_streak.empty:
        streak_a = ag.athlete_streak.copy()
        streak_b = ag.athlete_streak.rename(columns={"Nom": "Opponent", "Win_Streak": "B_Win_Streak"})
        d = d.merge(streak_a, on="Nom", how="left")
        d = d.merge(streak_b, on="Opponent", how="left")
        d["Win_Streak"] = d["Win_Streak"].fillna(0)
        d["B_Win_Streak"] = d["B_Win_Streak"].fillna(0)
    else:
        d["Win_Streak"] = 0.0
        d["B_Win_Streak"] = 0.0
    d["Win_Streak_Diff"] = (d["Win_Streak"] - d["B_Win_Streak"]) / 10.0

    # Time weight for sample weighting (more recent competitions get more weight)
    d["_time_weight"] = d.apply(
        lambda row: compute_time_weight(str(row.get("Competition", "")), row.get("Year")),
        axis=1,
    )

    d["y"] = d["Athlete_Win"].astype(int)

    out = d[FEATURE_COLS + ["y", "_time_weight"]].copy()
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


@st.cache_data(show_spinner=False)
def _train_model_and_aggs(df_in: pd.DataFrame) -> Tuple[Pipeline, Aggregates, np.ndarray, np.ndarray]:
    matches = _build_paired_matches(df_in)
    directed = _to_directed_rows(matches)

    ag = _compute_aggregates(directed)
    train = _prepare_training_table(directed, ag)

    pipe = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=800, C=0.5, solver="lbfgs", class_weight="balanced")),
    ])

    cv_accuracy = np.array([])
    cv_auc = np.array([])

    if train.empty:
        X = np.zeros((10, len(FEATURE_COLS)))
        y = np.array([0, 1] * 5)
        pipe.fit(X, y)
        return pipe, ag, cv_accuracy, cv_auc

    X = train[FEATURE_COLS].values
    y = train["y"].values

    # Time-based sample weighting: more recent competitions get higher weight
    sample_weights = train["_time_weight"].values if "_time_weight" in train.columns else np.ones(len(train))
    pipe.fit(X, y, clf__sample_weight=sample_weights)

    n_cv = min(5, len(train))
    if len(train) >= 10 and len(np.unique(y)) > 1:
        try:
            cv_accuracy = cross_val_score(pipe, X, y, cv=n_cv, scoring="accuracy")
        except Exception:
            cv_accuracy = np.array([])
        try:
            cv_auc = cross_val_score(pipe, X, y, cv=n_cv, scoring="roc_auc")
        except Exception:
            cv_auc = np.array([])

    return pipe, ag, cv_accuracy, cv_auc


# ═══════════════════════════════════════════════════════════════════════════════
# Kata selection & shrinkage — v2 (slower curve)
# ═══════════════════════════════════════════════════════════════════════════════

def _get_katas_of_style_in_scope(df_scope, style_a):
    katas_scope = df_scope["Kata"].dropna().astype(str).unique().tolist()
    if style_a is None or "Style" not in df_scope.columns:
        return sorted(set(map(str, katas_scope)))

    tmp = df_scope[["Kata", "Style"]].dropna()
    katas_style = sorted(tmp[tmp["Style"].astype(str) == str(style_a)]["Kata"].astype(str).unique().tolist())

    if "Suparinpei" in katas_scope and "Suparinpei" not in katas_style:
        katas_style.append("Suparinpei")
    return sorted(set(katas_style))


def _count_matches_for_athlete(matches: pd.DataFrame, nom: str) -> int:
    if matches.empty:
        return 0
    return int(((matches["Red_Nom"] == nom) | (matches["Blue_Nom"] == nom)).sum())


def _shrink_weight(a_matches, b_matches, a_kata_n):
    """✅ FIX: slower exponential curve (divisor 10/8 instead of 6/4)."""
    w_a = 1.0 - np.exp(-a_matches / 10.0)
    w_b = 1.0 - np.exp(-b_matches / 10.0)
    w_k = 1.0 - np.exp(-a_kata_n / 8.0)
    w = 0.40 * ((w_a + w_b) / 2.0) + 0.60 * w_k
    return float(np.clip(w, 0.08, 0.92))


def _compute_features_for_pair(df_scope, ag, matches_scope, selected_type_compet, nom_a, nom_b, style_a, style_b):
    nation_a = safe_mode(df_scope[df_scope["Nom"] == nom_a]["Nation"], default="")
    nation_b = safe_mode(df_scope[df_scope["Nom"] == nom_b]["Nation"], default="")
    same_nation = int(str(nation_a) == str(nation_b))
    same_style = int(str(style_a) == str(style_b))

    a_win = b_win = 0.5
    a_note_mean = b_note_mean = ag.global_note_mean
    a_ranking = b_ranking = 100.0
    if not ag.athlete_stats.empty:
        stats_map = ag.athlete_stats.set_index("Nom")
        if nom_a in stats_map.index:
            row_a = stats_map.loc[nom_a]
            a_win = float(row_a["WinRate_Smoothed"])
            a_note_mean = float(row_a["Note_Mean"]) if pd.notna(row_a["Note_Mean"]) else ag.global_note_mean
            a_ranking = float(row_a["Ranking_Mean"]) if pd.notna(row_a["Ranking_Mean"]) else 100.0
        if nom_b in stats_map.index:
            row_b = stats_map.loc[nom_b]
            b_win = float(row_b["WinRate_Smoothed"])
            b_note_mean = float(row_b["Note_Mean"]) if pd.notna(row_b["Note_Mean"]) else ag.global_note_mean
            b_ranking = float(row_b["Ranking_Mean"]) if pd.notna(row_b["Ranking_Mean"]) else 100.0

    winrate_adv = a_win - b_win
    ranking_adv = (b_ranking - a_ranking) / 100.0

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

    h2h_adv = _center_rate(h2h_wr_a) * np.log1p(h2h_total)
    is_k1 = 1 if selected_type_compet == "Premier League (K1)" else 0

    a_matches = _count_matches_for_athlete(matches_scope, nom_a)
    b_matches = _count_matches_for_athlete(matches_scope, nom_b)

    # v3 — new feature values for prediction
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

    # Recent win rate (momentum)
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

    # Experience (total matches)
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

    # Continent of A (for Is_Home)
    continent_a = safe_mode(df_scope[df_scope["Nom"] == nom_a]["Continent"], default="")
    region_a = safe_mode(df_scope[df_scope["Nom"] == nom_a]["Region_monde"], default="")

    # v4 — Flag WR
    a_flag_wr = b_flag_wr = 0.5
    if not ag.athlete_flag_stats.empty:
        f_map = ag.athlete_flag_stats.set_index("Nom")
        if nom_a in f_map.index:
            a_flag_wr = float(f_map.loc[nom_a, "Flag_WinRate"])
        if nom_b in f_map.index:
            b_flag_wr = float(f_map.loc[nom_b, "Flag_WinRate"])

    # v4 — Recent 5
    a_recent5 = b_recent5 = 0.5
    if not ag.athlete_recent_5.empty:
        r5_map = ag.athlete_recent_5.set_index("Nom")
        if nom_a in r5_map.index:
            a_recent5 = float(r5_map.loc[nom_a, "Recent5_WinRate"])
        if nom_b in r5_map.index:
            b_recent5 = float(r5_map.loc[nom_b, "Recent5_WinRate"])

    # v4 — Win streak
    a_streak = b_streak = 0.0
    if not ag.athlete_streak.empty:
        s_map = ag.athlete_streak.set_index("Nom")
        if nom_a in s_map.index:
            a_streak = float(s_map.loc[nom_a, "Win_Streak"])
        if nom_b in s_map.index:
            b_streak = float(s_map.loc[nom_b, "Win_Streak"])

    return {
        "same_nation": same_nation, "same_style": same_style, "is_k1": is_k1,
        "winrate_adv": winrate_adv, "h2h_adv": h2h_adv,
        "ranking_adv": ranking_adv,
        "a_note_mean": a_note_mean, "b_note_mean": b_note_mean,
        "a_matches": a_matches, "b_matches": b_matches,
        # v3
        "note_trend_diff": a_note_trend - b_note_trend,
        "note_std_diff": a_note_std - b_note_std,
        "momentum_diff": a_recent_wr - b_recent_wr,
        "age_diff": (age_a - age_b) / 10.0,
        "a_fav_kata": a_fav_kata,
        "exp_diff": float(np.log1p(a_total_m) - np.log1p(b_total_m)),
        "kata_diversity_diff": a_kata_div - b_kata_div,
        "continent_a": str(continent_a), "region_a": str(region_a),
        # v4
        "is_red": 1,
        "flag_winrate_diff": a_flag_wr - b_flag_wr,
        "recent_5_diff": a_recent5 - b_recent5,
        "win_streak_diff": (a_streak - b_streak) / 10.0,
    }


def _predict_for_katas(model, ag, nom_a, nom_b, n_tour, katas, base_feats):
    ok_index = ag.athlete_oppkata_losses.set_index(["Nom", "Opp_Kata"]) if not ag.athlete_oppkata_losses.empty else None
    ak_index = ag.athlete_kata.set_index(["Nom", "Kata"]) if not ag.athlete_kata.empty else None
    kt_index = ag.kata_tour.set_index(["Kata", "N_Tour"]) if not ag.kata_tour.empty else None
    ke_map = ag.kata_effect.set_index("Kata")["Kata_Effect"].to_dict() if not ag.kata_effect.empty else {}
    seen_index = ag.opponent_seen_kata.set_index(["Nom", "Opp_Kata"]) if not ag.opponent_seen_kata.empty else None

    # Competition region for Is_Home (approximate from last comp in scope)
    tour_rank = _tour_to_rank(n_tour) / 6.0

    results = []
    for kata in katas:
        kata = str(kata)

        if ak_index is not None and (nom_a, kata) in ak_index.index:
            row = ak_index.loc[(nom_a, kata)]
            a_kata_wr = float(row["Kata_WinRate_Smoothed"])
            a_kata_note = float(row["Kata_Note_Mean"]) if pd.notna(row["Kata_Note_Mean"]) else ag.global_note_mean
            a_kata_n = int(row["Kata_Total"])
        else:
            a_kata_wr = 0.5
            a_kata_note = ag.global_note_mean
            a_kata_n = 0

        b_loss_vs = 0.5
        if ok_index is not None and (nom_b, kata) in ok_index.index:
            b_loss_vs = float(ok_index.loc[(nom_b, kata), "LossRate_vs_Kata_Smoothed"])

        kata_tour_wr = 0.5
        if kt_index is not None and (kata, str(n_tour)) in kt_index.index:
            kata_tour_wr = float(kt_index.loc[(kata, str(n_tour)), "Kata_Tour_WinRate_Smoothed"])

        note_diff = a_kata_note - base_feats["b_note_mean"]
        a_kata_note_c = a_kata_note - ag.global_note_mean
        log_a_kata_n = float(np.log1p(a_kata_n))

        # v3 — per-kata features
        is_fav = int(base_feats["a_fav_kata"] == kata) if base_feats["a_fav_kata"] else 0
        b_seen = 0
        if seen_index is not None and (nom_b, kata) in seen_index.index:
            b_seen = int(seen_index.loc[(nom_b, kata), "Seen_Count"] >= 1)
        is_home = int(str(base_feats["continent_a"]) == str(base_feats["region_a"]))
        is_red = int(base_feats.get("is_red", 1))
        flag_wr_diff = float(base_feats.get("flag_winrate_diff", 0.0))

        X = np.array([[
            base_feats["same_nation"], base_feats["same_style"], base_feats["is_k1"],
            base_feats["h2h_adv"], base_feats["winrate_adv"],
            _center_rate(a_kata_wr), _center_rate(b_loss_vs),
            _center_rate(kata_tour_wr), float(ke_map.get(kata, 0.0)),
            note_diff, base_feats["ranking_adv"], log_a_kata_n, a_kata_note_c,
            # v3 features
            base_feats["note_trend_diff"], base_feats["momentum_diff"],
            base_feats["age_diff"], is_fav,
            base_feats["note_std_diff"], base_feats["exp_diff"],
            base_feats["kata_diversity_diff"], b_seen,
            is_home, tour_rank,
            # v4 features
            is_red, flag_wr_diff,
            base_feats["recent_5_diff"], base_feats["win_streak_diff"],
        ]], dtype=float)

        p_model = float(model.predict_proba(X)[0, 1])
        w = _shrink_weight(int(base_feats["a_matches"]), int(base_feats["b_matches"]), int(a_kata_n))
        p_final = float(np.clip(0.5 + w * (p_model - 0.5), 0.01, 0.99))

        results.append({
            "Kata": kata,
            "Probabilité de victoire (%)": round(p_final * 100.0, 2),
            "Confiance (0-1)": round(w, 3),
            "Note moy. A (kata)": round(a_kata_note, 2),
            "Diff. notes (A-B)": round(note_diff, 2),
            "Nb occ. (A, kata)": int(a_kata_n),
            "Nb matchs (A)": int(base_feats["a_matches"]),
            "Nb matchs (B)": int(base_feats["b_matches"]),
            "Kata favori A": "✓" if is_fav else "",
            "B connaît kata": "✓" if b_seen else "",
        })

    return pd.DataFrame(results).sort_values("Probabilité de victoire (%)", ascending=False).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Streamlit tab
# ═══════════════════════════════════════════════════════════════════════════════

@st.fragment
def show_proba_victoire_kata_tab(data: pd.DataFrame) -> None:
    st.header(t("Probabilité de victoire par kata"))
    show_tab_help("proba_victoire")

    if get_lang() == "en":
        st.markdown(
            """
**Warning**
- Results are **probabilities**, not absolute truth.
- Anti-bias: if A/B have little history, probability is **pulled toward 50%** (shrinkage).
- Model v4 integrates **27 features**: scores, ranking, head-to-head, temporal trend, recent momentum (15 & 5 matches), win streak, flag-era performance, age, favourite kata, consistency, experience, kata diversity, opponent familiarity, geographic advantage, belt color and round.
            """
        )
    else:
        st.markdown(
            """
**Avertissement**
- Les résultats sont des **probabilités**, pas une vérité absolue.
- Anti-biais : si A/B ont peu d'historique, on **ramène la proba vers 50%** (shrinkage).
- Le modèle v4 intègre **27 features** : notes, ranking, H2H, tendance temporelle, momentum récent (15 & 5 matchs), série en cours, performance drapeaux (2026+), âge, kata favori, consistance, expérience, diversité kata, familiarité adversaire, avantage géographique, couleur de ceinture et tour.
            """
        )

    df = data.copy()

    if "Note" in df.columns and df["Note"].dtype == object:
        df["Note"] = df["Note"].astype(str).str.replace(",", ".", regex=False)
    for col in ["Year", "Note", "Ranking"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    model, ag, cv_accuracy, cv_auc = _train_model_and_aggs(df)

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

        nom_a = st.selectbox(t("Athlète A"), athlete_names, key="proba_nom_a")

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

        nom_b = st.selectbox(t("Athlète B (adversaire)"), athlete_b_names, key="proba_nom_b")

        tour_options = sorted(df_scope["N_Tour"].dropna().astype(str).unique().tolist())
        if not tour_options:
            st.warning(t("Aucun tour disponible."))
            filter_panel_close()
            return

        n_tour = st.selectbox(t("Tour simulé"), tour_options, format_func=fmt_tour, key="proba_tour")

        style_a = safe_mode(df_scope[df_scope["Nom"] == nom_a]["Style"], default=None)
        style_b = safe_mode(df_scope[df_scope["Nom"] == nom_b]["Style"], default=None)

        st.markdown("---")
        st.markdown(t("#### 🥋 Katas testés (pour A)"))

        katas_style = _get_katas_of_style_in_scope(df_scope, style_a)
        katas_effectues_a = sorted(df_scope[df_scope["Nom"] == nom_a]["Kata"].dropna().astype(str).unique().tolist())

        katas_selectionnes = st.multiselect(
            t("Katas à tester"), options=katas_style,
            default=katas_effectues_a if katas_effectues_a else katas_style,
            key="proba_katas",
        )

        st.markdown("---")
        st.markdown(t("#### 🏆 Top 3 katas à faire"))
        if get_lang() == "en":
            st.caption(
                "Based on the opponent and round.\n\n"
                "- If A has **≥ 4 matches** in scope ⇒ Top 3 from their **already played katas**.\n"
                "- Otherwise ⇒ Top 3 from **all style katas**."
            )
        else:
            st.caption(
                "Basé sur l'adversaire et le tour.\n\n"
                "- Si A a **≥ 4 matchs** dans le scope ⇒ Top 3 parmi ses **katas déjà joués**.\n"
                "- Sinon ⇒ Top 3 parmi **tous les katas du style**."
            )
        run_top3 = st.button(t("🏆 Top 3"), key="proba_top3")

        filter_panel_close()

    with content_col:
        st.subheader(t("Comparaison sélectionnée"))

        # Quick stats cards
        c1, c2 = st.columns(2)
        with c1:
            a_stats_row = ag.athlete_stats[ag.athlete_stats["Nom"] == nom_a]
            if not a_stats_row.empty:
                r = a_stats_row.iloc[0]
                st.metric(t("A – Note moy."), f"{r['Note_Mean']:.1f}" if pd.notna(r["Note_Mean"]) else "?",
                          help=t("Note moyenne de l'athlète A sur tous ses passages"))
                st.caption(f"Win rate : {r['WinRate_Smoothed']:.0%} ({int(r['Wins'])}/{int(r['Total'])})")
                st.caption(f"{t('Ranking moy.')} : {r['Ranking_Mean']:.0f}" if pd.notna(r["Ranking_Mean"]) else "")
            else:
                st.metric(t("A – Note moy."), "?")
        with c2:
            b_stats_row = ag.athlete_stats[ag.athlete_stats["Nom"] == nom_b]
            if not b_stats_row.empty:
                r = b_stats_row.iloc[0]
                st.metric(t("B – Note moy."), f"{r['Note_Mean']:.1f}" if pd.notna(r["Note_Mean"]) else "?",
                          help=t("Note moyenne de l'athlète B sur tous ses passages"))
                st.caption(f"Win rate : {r['WinRate_Smoothed']:.0%} ({int(r['Wins'])}/{int(r['Total'])})")
                st.caption(f"{t('Ranking moy.')} : {r['Ranking_Mean']:.0f}" if pd.notna(r["Ranking_Mean"]) else "")
            else:
                st.metric(t("B – Note moy."), "?")

        st.markdown(
            f"**{t('Tour')} :** {n_tour} · **{t('Périmètre')} :** {selected_type_compet}"
        )

        matches_scope = _build_paired_matches(df_scope)

        base_feats = _compute_features_for_pair(
            df_scope=df_scope, ag=ag, matches_scope=matches_scope,
            selected_type_compet=selected_type_compet,
            nom_a=nom_a, nom_b=nom_b,
            style_a=style_a if style_a != "Non spécifié" else None,
            style_b=style_b if style_b != "Non spécifié" else None,
        )

        st.markdown(
            f"📌 {t('Historique scope')} : **A={base_feats['a_matches']} {t('matchs')}**, **B={base_feats['b_matches']} {t('matchs')}**"
        )

        # Model performance
        with st.expander(t("📊 Performance du modèle (cross-validation)")):
            if len(cv_accuracy) > 0:
                st.write(f"**Accuracy (CV)** : {cv_accuracy.mean():.3f} ± {cv_accuracy.std():.3f}")
            else:
                st.write(t("Accuracy (CV) : données insuffisantes"))
            if len(cv_auc) > 0:
                st.write(f"**AUC (CV)** : {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")
            else:
                st.write(t("AUC (CV) : données insuffisantes"))

            st.markdown(t("##### Feature importances (coefficients)"))
            _FEATURE_LABELS = {
                "Same_Nation": t("Même nationalité"),
                "Same_Style": t("Même style de kata"),
                "Is_K1": t("Compétition K1 (vs SA)"),
                "H2H_Adv": t("Avantage face-à-face historique"),
                "WinRate_Adv": t("Avantage win rate global"),
                "A_Kata_WinRate_C": t("Win rate de A avec ce kata"),
                "B_Weakness_C": t("Faiblesse de B face à ce kata"),
                "Kata_Tour_C": t("Efficacité du kata dans ce tour"),
                "Kata_Effect": t("Effet résiduel du kata"),
                "Note_Diff": t("Différence de notes (A-B)"),
                "Ranking_Adv": t("Avantage au classement mondial"),
                "Log_A_Kata_N": t("Expérience de A avec ce kata"),
                "A_Kata_Note_C": t("Note de A avec ce kata (vs moyenne)"),
                "Note_Trend_Diff": t("Différence de tendance de notes"),
                "Momentum_Diff": t("Différence de dynamique récente (15 matchs)"),
                "Age_Diff": t("Différence d'âge"),
                "Is_Fav_Kata_A": t("A joue son kata favori"),
                "Note_Std_Diff": t("Différence de régularité"),
                "Exp_Diff": t("Différence d'expérience (nb matchs)"),
                "Kata_Diversity_Diff": t("Différence de diversité kata"),
                "B_Seen_Kata": t("B a déjà vu ce kata"),
                "Is_Home": t("A joue à domicile (continent)"),
                "Tour_Rank": t("Phase de compétition (poule → finale)"),
                "Is_Red": t("A porte la ceinture rouge"),
                "Flag_WinRate_Diff": t("Avantage win rate drapeaux (2026+)"),
                "Recent_5_Diff": t("Momentum court terme (5 derniers matchs)"),
                "Win_Streak_Diff": t("Différence de série en cours (victoires/défaites)"),
            }
            try:
                coefs = model.named_steps["clf"].coef_[0]
                fi_df = pd.DataFrame({"Feature": FEATURE_COLS, "Coefficient": coefs})
                fi_df[t("Explication")] = fi_df["Feature"].map(_FEATURE_LABELS).fillna(fi_df["Feature"])
                fi_df["Abs"] = fi_df["Coefficient"].abs()
                fi_df = fi_df.sort_values("Abs", ascending=False).drop(columns=["Abs"])
                st.dataframe(format_display_df(fi_df[[t("Explication"), "Feature", "Coefficient"]]), width="stretch")
            except Exception:
                st.write(t("Impossible d'extraire les coefficients."))

        if not katas_selectionnes:
            st.info(t("Sélectionne au moins un kata à tester."))

        run_manual = st.button(t("Calculer les probabilités (katas sélectionnés)"), key="proba_run")

        if run_manual and katas_selectionnes:
            res_df = _predict_for_katas(
                model=model, ag=ag, nom_a=nom_a, nom_b=nom_b,
                n_tour=str(n_tour), katas=list(map(str, katas_selectionnes)),
                base_feats=base_feats,
            )

            st.subheader(f"{t('Résultats')} – {nom_a} vs {nom_b} ({t('tour')}: {fmt_tour(n_tour)})")

            # Barres de probabilité colorées pour le Top 3
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
            st.dataframe(res_df, width="stretch")

            fig = px.bar(
                res_df, x="Kata", y="Probabilité de victoire (%)",
                color="Probabilité de victoire (%)",
                color_continuous_scale=["#dc3545", "#ffc107", "#28a745"],
                range_color=[30, 70],
                hover_data=["Confiance (0-1)", "Diff. notes (A-B)", "Nb occ. (A, kata)"],
                title=t("Probabilité de victoire estimée par kata (A) – sélection"),
            )
            st.plotly_chart(fig, width="stretch", key="proba_kata_bar_manual")

            st.info(t(
                "✅ Modèle v3 — 22 features : notes, ranking, H2H, tendance temporelle, "
                "momentum, âge, kata favori, consistance, expérience, diversité, "
                "familiarité adversaire, avantage géo, tour."
            ))

        if run_top3:
            a_match_count = int(base_feats["a_matches"])
            if a_match_count >= 4:
                candidates = sorted(set(df_scope[df_scope["Nom"] == nom_a]["Kata"].dropna().astype(str).unique().tolist()))
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
                st.dataframe(res_df_top.head(3), width="stretch")

                fig2 = px.bar(
                    res_df_top.head(10), x="Kata", y="Probabilité de victoire (%)",
                    hover_data=["Confiance (0-1)", "Diff. notes (A-B)", "Nb occ. (A, kata)"],
                    title=t("Top katas recommandés (Top 10 affiché)"),
                )
                st.plotly_chart(fig2, width="stretch", key="proba_kata_bar_top3")
