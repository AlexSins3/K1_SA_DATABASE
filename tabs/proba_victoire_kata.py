# tabs/proba_victoire_kata.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =========================================================
# Utilitaires
# =========================================================

def _to_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def _normalize_bool_to_int(v) -> int | None:
    if pd.isna(v):
        return None
    if isinstance(v, (bool, np.bool_)):
        return int(bool(v))

    s = str(v).strip().lower()
    if s in {"true", "vrai", "1", "oui", "y", "yes", "win", "gagné", "gagne"}:
        return 1
    if s in {"false", "faux", "0", "non", "n", "no", "lose", "perdu"}:
        return 0
    return int(bool(v))


def _safe_mode(series: pd.Series, default="Non spécifié"):
    m = series.dropna()
    if m.empty:
        return default
    try:
        return m.mode().iloc[0]
    except Exception:
        return m.iloc[0]


def _beta_smooth_rate(wins: float, total: float, alpha: float = 2.0, beta: float = 2.0) -> float:
    return (wins + alpha) / (total + alpha + beta)


def _center_rate(x: float) -> float:
    try:
        return float(x) - 0.5
    except Exception:
        return 0.0


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


# =========================================================
# Pairing R/B -> matches
# =========================================================

def _build_paired_matches(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy().reset_index(drop=True)

    needed = ["Nom", "Ceinture", "Kata", "N_Tour", "Competition", "Type_Compet", "Victoire"]
    for c in needed:
        if c not in d.columns:
            raise ValueError(f"Colonne manquante: {c}")

    if "Year" not in d.columns:
        d["Year"] = np.nan
    if "Nation" not in d.columns:
        d["Nation"] = np.nan
    if "Style" not in d.columns:
        d["Style"] = np.nan
    if "Sexe" not in d.columns:
        d["Sexe"] = np.nan
    if "Note" not in d.columns:
        d["Note"] = np.nan

    rows = []
    n = len(d)

    def same_context(r1, r2) -> bool:
        return (
            (str(r1.get("Competition")) == str(r2.get("Competition")))
            and (str(r1.get("Type_Compet")) == str(r2.get("Type_Compet")))
            and (str(r1.get("N_Tour")) == str(r2.get("N_Tour")))
            and (str(r1.get("Year")) == str(r2.get("Year")))
        )

    for i in range(n - 1):
        r1 = d.iloc[i]
        r2 = d.iloc[i + 1]

        c1 = str(r1.get("Ceinture"))
        c2 = str(r2.get("Ceinture"))

        if {"R", "B"}.issubset({c1, c2}) and same_context(r1, r2):
            if c1 == "R" and c2 == "B":
                red = r1
                blue = r2
            else:
                red = r2
                blue = r1

            y_red = _normalize_bool_to_int(red.get("Victoire"))
            if y_red is None:
                continue

            rows.append(
                {
                    "Competition": red.get("Competition"),
                    "Year": red.get("Year"),
                    "Type_Compet": red.get("Type_Compet"),
                    "N_Tour": str(red.get("N_Tour")),
                    "Red_Nom": red.get("Nom"),
                    "Blue_Nom": blue.get("Nom"),
                    "Red_Kata": red.get("Kata"),
                    "Blue_Kata": blue.get("Kata"),
                    "Red_Nation": red.get("Nation"),
                    "Blue_Nation": blue.get("Nation"),
                    "Red_Style": red.get("Style"),
                    "Blue_Style": blue.get("Style"),
                    "Red_Sexe": red.get("Sexe"),
                    "Blue_Sexe": blue.get("Sexe"),
                    "Red_Note": _to_float(str(red.get("Note")).replace(",", ".")),
                    "Blue_Note": _to_float(str(blue.get("Note")).replace(",", ".")),
                    "Red_Win": int(y_red),
                }
            )

    return pd.DataFrame(rows)


def _to_directed_rows(matches: pd.DataFrame) -> pd.DataFrame:
    if matches.empty:
        return pd.DataFrame()

    rows = []
    for _, m in matches.iterrows():
        rows.append(
            {
                "Nom": m["Red_Nom"],
                "Opponent": m["Blue_Nom"],
                "Kata": m["Red_Kata"],
                "Opp_Kata": m["Blue_Kata"],
                "N_Tour": str(m["N_Tour"]),
                "Competition": m["Competition"],
                "Year": m["Year"],
                "Type_Compet": m["Type_Compet"],
                "Nation": m["Red_Nation"],
                "Opp_Nation": m["Blue_Nation"],
                "Style": m["Red_Style"],
                "Opp_Style": m["Blue_Style"],
                "Sexe": m["Red_Sexe"],
                "Opp_Sexe": m["Blue_Sexe"],
                "Note": m["Red_Note"],
                "Athlete_Win": int(m["Red_Win"]),
            }
        )
        rows.append(
            {
                "Nom": m["Blue_Nom"],
                "Opponent": m["Red_Nom"],
                "Kata": m["Blue_Kata"],
                "Opp_Kata": m["Red_Kata"],
                "N_Tour": str(m["N_Tour"]),
                "Competition": m["Competition"],
                "Year": m["Year"],
                "Type_Compet": m["Type_Compet"],
                "Nation": m["Blue_Nation"],
                "Opp_Nation": m["Red_Nation"],
                "Style": m["Blue_Style"],
                "Opp_Style": m["Red_Style"],
                "Sexe": m["Blue_Sexe"],
                "Opp_Sexe": m["Red_Sexe"],
                "Note": m["Blue_Note"],
                "Athlete_Win": int(1 - int(m["Red_Win"])),
            }
        )

    return pd.DataFrame(rows)


# =========================================================
# Agrégats
# =========================================================

@dataclass
class Aggregates:
    athlete_stats: pd.DataFrame
    athlete_kata: pd.DataFrame
    athlete_oppkata_losses: pd.DataFrame
    kata_tour: pd.DataFrame
    h2h: pd.DataFrame
    kata_effect: pd.DataFrame


def _compute_aggregates(directed: pd.DataFrame) -> Aggregates:
    if directed.empty:
        empty = pd.DataFrame()
        return Aggregates(empty, empty, empty, empty, empty, empty)

    d = directed.copy()

    # overall athlete winrate (smoothed) + Total (important pour la confiance)
    a = d.groupby("Nom")["Athlete_Win"].agg(["sum", "count"]).reset_index()
    a.rename(columns={"sum": "Wins", "count": "Total"}, inplace=True)
    a["WinRate_Smoothed"] = a.apply(lambda r: _beta_smooth_rate(r["Wins"], r["Total"], 2, 2), axis=1)

    # athlete-kata
    ak = d.groupby(["Nom", "Kata"])["Athlete_Win"].agg(["sum", "count"]).reset_index()
    ak.rename(columns={"sum": "Kata_Wins", "count": "Kata_Total"}, inplace=True)
    notes = d.dropna(subset=["Note"]).groupby(["Nom", "Kata"])["Note"].mean().reset_index()
    notes.rename(columns={"Note": "Kata_Note_Mean"}, inplace=True)
    ak = ak.merge(notes, on=["Nom", "Kata"], how="left")
    ak["Kata_WinRate_Smoothed"] = ak.apply(
        lambda r: _beta_smooth_rate(r["Kata_Wins"], r["Kata_Total"], 1.5, 1.5), axis=1
    )

    # opponent weakness vs kata (loss rate vs Opp_Kata)
    d["Loss"] = 1 - d["Athlete_Win"]
    ok = d.groupby(["Nom", "Opp_Kata"])["Loss"].agg(["sum", "count"]).reset_index()
    ok.rename(columns={"sum": "Losses_vs_Kata", "count": "Total_vs_Kata"}, inplace=True)
    ok["LossRate_vs_Kata_Smoothed"] = ok.apply(
        lambda r: _beta_smooth_rate(r["Losses_vs_Kata"], r["Total_vs_Kata"], 1.5, 1.5), axis=1
    )

    # kata-tour win rate
    kt = d.groupby(["Kata", "N_Tour"])["Athlete_Win"].agg(["sum", "count"]).reset_index()
    kt.rename(columns={"sum": "Wins", "count": "Total"}, inplace=True)
    kt["Kata_Tour_WinRate_Smoothed"] = kt.apply(lambda r: _beta_smooth_rate(r["Wins"], r["Total"], 2, 2), axis=1)

    # h2h
    tmp = d[["Nom", "Opponent", "Athlete_Win"]].copy()
    tmp["A"] = tmp.apply(lambda r: min(str(r["Nom"]), str(r["Opponent"])), axis=1)
    tmp["B"] = tmp.apply(lambda r: max(str(r["Nom"]), str(r["Opponent"])), axis=1)
    tmp["A_is_Nom"] = tmp["Nom"] == tmp["A"]
    tmp["Win_for_A"] = np.where(tmp["A_is_Nom"], tmp["Athlete_Win"], 1 - tmp["Athlete_Win"])
    h2h = tmp.groupby(["A", "B"])["Win_for_A"].agg(["sum", "count"]).reset_index()
    h2h.rename(columns={"sum": "Wins_A", "count": "Total"}, inplace=True)
    h2h["WinRate_A_Smoothed"] = h2h.apply(lambda r: _beta_smooth_rate(r["Wins_A"], r["Total"], 1.5, 1.5), axis=1)

    # kata effect anti-biais (résiduel vs force de base)
    a_map = a.set_index("Nom")["WinRate_Smoothed"].to_dict()
    d["Athlete_Base"] = d["Nom"].map(a_map).fillna(0.5)
    d["Residual_vs_Base"] = d["Athlete_Win"] - d["Athlete_Base"]

    ke = d.groupby("Kata")["Residual_vs_Base"].agg(["mean", "count"]).reset_index()
    ke.rename(columns={"mean": "Residual_Mean", "count": "Kata_Uses"}, inplace=True)
    k = 25.0
    ke["Kata_Effect"] = ke["Residual_Mean"] * (ke["Kata_Uses"] / (ke["Kata_Uses"] + k))

    return Aggregates(a, ak, ok, kt, h2h, ke)


# =========================================================
# Modèle : features centrées (0 = neutre)
# =========================================================

FEATURE_COLS = [
    "Same_Nation",
    "Same_Style",
    "Is_K1",
    "H2H_Adv",          # (wr-0.5)*log1p(total)
    "WinRate_Adv",      # A_win - B_win
    "A_Kata_WinRate_C", # A_kata_wr - 0.5
    "B_Weakness_C",     # B_loss_vs_kata - 0.5
    "Kata_Tour_C",      # kata_tour_wr - 0.5
    "Kata_Effect",
    "A_Kata_Note",
]


def _prepare_training_table(directed: pd.DataFrame, ag: Aggregates) -> pd.DataFrame:
    if directed.empty:
        return pd.DataFrame()

    d = directed.copy()
    d["Same_Nation"] = (d["Nation"].astype(str) == d["Opp_Nation"].astype(str)).astype(int)
    d["Same_Style"] = (d["Style"].astype(str) == d["Opp_Style"].astype(str)).astype(int)
    d["Is_K1"] = (d["Type_Compet"].astype(str) == "K1").astype(int)

    # overall win rates
    a_map = ag.athlete_stats.set_index("Nom")["WinRate_Smoothed"].to_dict()
    d["A_WinRate"] = d["Nom"].map(a_map).fillna(0.5)
    d["B_WinRate"] = d["Opponent"].map(a_map).fillna(0.5)
    d["WinRate_Adv"] = d["A_WinRate"] - d["B_WinRate"]

    # athlete-kata
    ak = ag.athlete_kata.set_index(["Nom", "Kata"])
    d["A_Kata_WinRate"] = d.apply(
        lambda r: float(ak.loc[(r["Nom"], r["Kata"]), "Kata_WinRate_Smoothed"])
        if (r["Nom"], r["Kata"]) in ak.index else 0.5,
        axis=1,
    )
    d["A_Kata_WinRate_C"] = d["A_Kata_WinRate"].apply(_center_rate)

    d["A_Kata_Note"] = d.apply(
        lambda r: float(ak.loc[(r["Nom"], r["Kata"]), "Kata_Note_Mean"])
        if (r["Nom"], r["Kata"]) in ak.index and pd.notna(ak.loc[(r["Nom"], r["Kata"]), "Kata_Note_Mean"]) else 0.0,
        axis=1,
    )

    # B weakness vs A kata
    ok = ag.athlete_oppkata_losses.set_index(["Nom", "Opp_Kata"])
    d["B_LossRate_vs_AKata"] = d.apply(
        lambda r: float(ok.loc[(r["Opponent"], r["Kata"]), "LossRate_vs_Kata_Smoothed"])
        if (r["Opponent"], r["Kata"]) in ok.index else 0.5,
        axis=1,
    )
    d["B_Weakness_C"] = d["B_LossRate_vs_AKata"].apply(_center_rate)

    # kata-tour
    kt = ag.kata_tour.set_index(["Kata", "N_Tour"])
    d["Kata_Tour_WinRate"] = d.apply(
        lambda r: float(kt.loc[(r["Kata"], str(r["N_Tour"])), "Kata_Tour_WinRate_Smoothed"])
        if (r["Kata"], str(r["N_Tour"])) in kt.index else 0.5,
        axis=1,
    )
    d["Kata_Tour_C"] = d["Kata_Tour_WinRate"].apply(_center_rate)

    # kata effect
    ke = ag.kata_effect.set_index("Kata")["Kata_Effect"].to_dict()
    d["Kata_Effect"] = d["Kata"].map(ke).fillna(0.0)

    # head2head adv
    h = ag.h2h.copy()
    if not h.empty:
        h_index = h.set_index(["A", "B"])

        def get_h2h(nom: str, opp: str) -> Tuple[float, float]:
            a_ = min(str(nom), str(opp))
            b_ = max(str(nom), str(opp))
            if (a_, b_) not in h_index.index:
                return 0.0, 0.5
            row = h_index.loc[(a_, b_)]
            total = float(row["Total"])
            winrate_A = float(row["WinRate_A_Smoothed"])
            wr_nom = winrate_A if str(nom) == a_ else (1.0 - winrate_A)
            return total, wr_nom

        h_vals = d.apply(lambda r: get_h2h(r["Nom"], r["Opponent"]), axis=1)
        d["H2H_Total"] = [v[0] for v in h_vals]
        d["H2H_WR"] = [v[1] for v in h_vals]
    else:
        d["H2H_Total"] = 0.0
        d["H2H_WR"] = 0.5

    d["H2H_Adv"] = (d["H2H_WR"].apply(_center_rate)) * np.log1p(d["H2H_Total"])

    d["y"] = d["Athlete_Win"].astype(int)

    out = d[FEATURE_COLS + ["y"]].copy()
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


@st.cache_data(show_spinner=False)
def _train_model_and_aggs(df_in: pd.DataFrame) -> Tuple[Pipeline, Aggregates]:
    matches = _build_paired_matches(df_in)
    directed = _to_directed_rows(matches)

    ag = _compute_aggregates(directed)
    train = _prepare_training_table(directed, ag)

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=800, C=0.5, solver="lbfgs", class_weight="balanced")),
        ]
    )

    if train.empty:
        X = np.zeros((10, len(FEATURE_COLS)))
        y = np.array([0, 1] * 5)
        pipe.fit(X, y)
        return pipe, ag

    X = train[FEATURE_COLS].values
    y = train["y"].values
    pipe.fit(X, y)
    return pipe, ag


# =========================================================
# Katas / confiance / shrinkage final
# =========================================================

def _get_katas_of_style_in_scope(df_scope: pd.DataFrame, style_a: str | None) -> list[str]:
    katas_scope = df_scope["Kata"].dropna().astype(str).unique().tolist()

    if style_a is None or "Style" not in df_scope.columns:
        return sorted(set(map(str, katas_scope)))

    tmp = df_scope[["Kata", "Style"]].dropna()
    katas_style = sorted(
        tmp[tmp["Style"].astype(str) == str(style_a)]["Kata"].astype(str).unique().tolist()
    )

    if "Suparinpei" in katas_scope and "Suparinpei" not in katas_style:
        katas_style.append("Suparinpei")

    return sorted(set(katas_style))


def _count_matches_for_athlete_in_scope(df_scope: pd.DataFrame, nom: str) -> int:
    matches = _build_paired_matches(df_scope)
    if matches.empty:
        return 0
    return int(((matches["Red_Nom"] == nom) | (matches["Blue_Nom"] == nom)).sum())


def _kata_occ_for_athlete(ag: Aggregates, nom: str, kata: str) -> int:
    if ag.athlete_kata.empty:
        return 0
    sub = ag.athlete_kata[(ag.athlete_kata["Nom"] == nom) & (ag.athlete_kata["Kata"] == kata)]
    if sub.empty:
        return 0
    return int(sub["Kata_Total"].iloc[0])


def _shrink_weight(a_matches: int, b_matches: int, a_kata_n: int) -> float:
    """
    Poids de confiance w ∈ [0,1].
    - si A/B ont très peu de matchs, w chute => proba ramenée vers 50%
    - si A a déjà fait ce kata plusieurs fois, w augmente
    """
    # saturations douces
    w_a = 1.0 - np.exp(-a_matches / 6.0)    # ~0.15 à 1 match, ~0.63 à 6 matchs
    w_b = 1.0 - np.exp(-b_matches / 6.0)
    w_k = 1.0 - np.exp(-a_kata_n / 4.0)     # ~0.22 à 1 occ, ~0.63 à 4 occ

    # priorité : historique global du duel (A+B) puis kata
    w = 0.45 * ((w_a + w_b) / 2.0) + 0.55 * w_k

    # on impose un plancher pour éviter de tout coller à 50%,
    # mais on évite aussi les extrêmes sans data
    return float(np.clip(w, 0.10, 0.95))


def _compute_features_for_pair(
    df_scope: pd.DataFrame,
    ag: Aggregates,
    selected_type_compet: str,
    nom_a: str,
    nom_b: str,
    style_a: str | None,
    style_b: str | None,
) -> dict:
    nation_a = _safe_mode(df_scope[df_scope["Nom"] == nom_a]["Nation"], default="")
    nation_b = _safe_mode(df_scope[df_scope["Nom"] == nom_b]["Nation"], default="")
    same_nation = int(str(nation_a) == str(nation_b))
    same_style = int(str(style_a) == str(style_b))

    # winrate adv
    a_win = 0.5
    b_win = 0.5
    if not ag.athlete_stats.empty:
        a_map = ag.athlete_stats.set_index("Nom")["WinRate_Smoothed"].to_dict()
        a_win = float(a_map.get(nom_a, 0.5))
        b_win = float(a_map.get(nom_b, 0.5))
    winrate_adv = a_win - b_win

    # h2h adv
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
    is_k1 = 1 if selected_type_compet == "Premier League (K1)" else (0 if selected_type_compet == "Series A (SA)" else 0)

    # match counts (pour shrinkage final)
    a_matches = _count_matches_for_athlete_in_scope(df_scope, nom_a)
    b_matches = _count_matches_for_athlete_in_scope(df_scope, nom_b)

    return {
        "same_nation": same_nation,
        "same_style": same_style,
        "is_k1": is_k1,
        "winrate_adv": winrate_adv,
        "h2h_adv": h2h_adv,
        "a_matches": a_matches,
        "b_matches": b_matches,
    }


def _predict_for_katas(
    model: Pipeline,
    ag: Aggregates,
    nom_a: str,
    nom_b: str,
    n_tour: str,
    katas: list[str],
    base_feats: dict,
) -> pd.DataFrame:
    ok_index = ag.athlete_oppkata_losses.set_index(["Nom", "Opp_Kata"]) if not ag.athlete_oppkata_losses.empty else None
    ak_index = ag.athlete_kata.set_index(["Nom", "Kata"]) if not ag.athlete_kata.empty else None
    kt_index = ag.kata_tour.set_index(["Kata", "N_Tour"]) if not ag.kata_tour.empty else None
    ke_map = ag.kata_effect.set_index("Kata")["Kata_Effect"].to_dict() if not ag.kata_effect.empty else {}

    results = []
    for kata in katas:
        kata = str(kata)

        # A kata stats
        if ak_index is not None and (nom_a, kata) in ak_index.index:
            row = ak_index.loc[(nom_a, kata)]
            a_kata_wr = float(row["Kata_WinRate_Smoothed"])
            a_kata_note = float(row["Kata_Note_Mean"]) if pd.notna(row["Kata_Note_Mean"]) else 0.0
            a_kata_n = int(row["Kata_Total"])
        else:
            a_kata_wr = 0.5
            a_kata_note = 0.0
            a_kata_n = 0

        # B weakness vs that kata
        if ok_index is not None and (nom_b, kata) in ok_index.index:
            b_loss_vs = float(ok_index.loc[(nom_b, kata), "LossRate_vs_Kata_Smoothed"])
        else:
            b_loss_vs = 0.5

        # kata-tour
        if kt_index is not None and (kata, str(n_tour)) in kt_index.index:
            kata_tour_wr = float(kt_index.loc[(kata, str(n_tour)), "Kata_Tour_WinRate_Smoothed"])
        else:
            kata_tour_wr = 0.5

        kata_effect = float(ke_map.get(kata, 0.0))

        X = np.array(
            [[
                base_feats["same_nation"],
                base_feats["same_style"],
                base_feats["is_k1"],
                base_feats["h2h_adv"],
                base_feats["winrate_adv"],
                _center_rate(a_kata_wr),
                _center_rate(b_loss_vs),
                _center_rate(kata_tour_wr),
                kata_effect,
                a_kata_note,
            ]],
            dtype=float,
        )

        p_model = float(model.predict_proba(X)[0, 1])  # 0..1

        # ✅ shrinkage final (anti “1 match => 80%”)
        w = _shrink_weight(
            a_matches=int(base_feats["a_matches"]),
            b_matches=int(base_feats["b_matches"]),
            a_kata_n=int(a_kata_n),
        )
        p_final = 0.5 + w * (p_model - 0.5)
        p_final = float(np.clip(p_final, 0.01, 0.99))

        # score confiance affiché
        conf = float(np.round(w, 3))

        results.append(
            {
                "Kata": kata,
                "Probabilité de victoire (%)": round(p_final * 100.0, 3),
                "Confiance (0-1)": conf,
                "Nb occurrences (A, kata)": int(a_kata_n),
                "Nb matchs (A)": int(base_feats["a_matches"]),
                "Nb matchs (B)": int(base_feats["b_matches"]),
            }
        )

    return pd.DataFrame(results).sort_values("Probabilité de victoire (%)", ascending=False).reset_index(drop=True)


# =========================================================
# Onglet Streamlit
# =========================================================

def show_proba_victoire_kata_tab(data: pd.DataFrame) -> None:
    st.header("Probabilité de victoire par kata")

    st.markdown(
        """
**Avertissement**
- Les résultats sont des **probabilités**, pas une vérité absolue.
- Fix anti-biais : si A/B ont peu d’historique, on **ramène la proba vers 50%** (shrinkage).
        """
    )

    df = data.copy()

    if "Note" in df.columns and df["Note"].dtype == object:
        df["Note"] = df["Note"].astype(str).str.replace(",", ".", regex=False)
    for col in ["Year", "Note"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    model, ag = _train_model_and_aggs(df)

    filters_col, content_col = st.columns([0.9, 2.4])

    run_manual = False
    run_top3 = False

    with filters_col:
        st.markdown(
            """
            <style>
            .filter-panel {
                background-color: #f6f8ff;
                padding: 1rem 1.2rem;
                border-radius: 0.8rem;
                border: 1px solid #d9e1ff;
            }
            .filter-panel h3, .filter-panel h4 {
                margin-top: 0.2rem;
                margin-bottom: 0.6rem;
            }
            .filter-panel label { font-weight: 500; }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div class='filter-panel'>", unsafe_allow_html=True)
        st.markdown("### 🎯 Paramètres")

        type_compet_options = ["Tous", "Premier League (K1)", "Series A (SA)"]
        selected_type_compet = st.radio("Type de compétition", type_compet_options, key="proba_type_compet")

        df_scope = df.copy()
        if selected_type_compet == "Premier League (K1)":
            df_scope = df_scope[df_scope["Type_Compet"] == "K1"]
        elif selected_type_compet == "Series A (SA)":
            df_scope = df_scope[df_scope["Type_Compet"] == "SA"]

        athlete_names = sorted(df_scope["Nom"].dropna().unique().tolist())
        if not athlete_names:
            st.warning("Aucun athlète disponible avec ces filtres.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        nom_a = st.selectbox("Athlète A", athlete_names, key="proba_nom_a")

        sexe_a = _safe_mode(df_scope[df_scope["Nom"] == nom_a]["Sexe"], default=None)
        if sexe_a is not None:
            athlete_b_names = sorted(
                df_scope[(df_scope["Sexe"] == sexe_a) & (df_scope["Nom"] != nom_a)]["Nom"]
                .dropna().unique().tolist()
            )
        else:
            athlete_b_names = sorted(df_scope[df_scope["Nom"] != nom_a]["Nom"].dropna().unique().tolist())

        if not athlete_b_names:
            st.warning("Aucun adversaire compatible (même sexe) trouvé dans ce périmètre.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        nom_b = st.selectbox("Athlète B", athlete_b_names, key="proba_nom_b")

        tour_options = sorted(df_scope["N_Tour"].dropna().astype(str).unique().tolist())
        if not tour_options:
            st.warning("Aucun tour (N_Tour) disponible dans ce périmètre.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        n_tour = st.selectbox("Tour (N_Tour)", tour_options, key="proba_tour")

        style_a = _safe_mode(df_scope[df_scope["Nom"] == nom_a]["Style"], default=None)
        style_b = _safe_mode(df_scope[df_scope["Nom"] == nom_b]["Style"], default=None)

        st.markdown("---")
        st.markdown("#### 🥋 Katas testés (pour A)")

        katas_style = _get_katas_of_style_in_scope(df_scope, style_a)

        katas_effectues_a = sorted(df_scope[df_scope["Nom"] == nom_a]["Kata"].dropna().astype(str).unique().tolist())

        katas_selectionnes = st.multiselect(
            "Katas à tester",
            options=katas_style,
            default=katas_effectues_a if katas_effectues_a else katas_style,
            key="proba_katas",
        )

        st.markdown("---")
        st.markdown("#### 🏆 Top 3 katas à faire")
        st.caption(
            "Basé sur l’adversaire et le tour.\n\n"
            "- Si A a **≥ 4 matchs** dans le scope ⇒ Top 3 parmi ses **katas déjà joués**.\n"
            "- Sinon ⇒ Top 3 parmi **tous les katas du style**."
        )
        run_top3 = st.button("🏆 Proposer le Top 3", key="proba_top3")

        st.markdown("</div>", unsafe_allow_html=True)

    with content_col:
        st.subheader("Comparaison sélectionnée")
        st.markdown(
            f"""
- **A :** {nom_a} (style: {style_a if style_a is not None else "?"})
- **B :** {nom_b} (style: {style_b if style_b is not None else "?"})
- **Tour :** {n_tour}
- **Périmètre :** {selected_type_compet}
            """
        )

        base_feats = _compute_features_for_pair(
            df_scope=df_scope,
            ag=ag,
            selected_type_compet=selected_type_compet,
            nom_a=nom_a,
            nom_b=nom_b,
            style_a=style_a if style_a != "Non spécifié" else None,
            style_b=style_b if style_b != "Non spécifié" else None,
        )

        st.markdown(
            f"📌 Historique scope : **A={base_feats['a_matches']} matchs**, **B={base_feats['b_matches']} matchs**"
        )

        if not katas_selectionnes:
            st.info("Sélectionne au moins un kata à tester.")

        run_manual = st.button("Calculer les probabilités (katas sélectionnés)", key="proba_run")

        if run_manual and katas_selectionnes:
            res_df = _predict_for_katas(
                model=model,
                ag=ag,
                nom_a=nom_a,
                nom_b=nom_b,
                n_tour=str(n_tour),
                katas=list(map(str, katas_selectionnes)),
                base_feats=base_feats,
            )

            st.subheader(f"Résultats – {nom_a} vs {nom_b} (tour: {n_tour})")
            st.markdown("##### Top 3 (sur ta sélection)")
            st.dataframe(res_df.head(3), width="stretch")

            st.markdown("##### Détail complet")
            st.dataframe(res_df, width="stretch")

            fig = px.bar(
                res_df,
                x="Kata",
                y="Probabilité de victoire (%)",
                hover_data=["Confiance (0-1)", "Nb occurrences (A, kata)", "Nb matchs (A)", "Nb matchs (B)"],
                title="Probabilité de victoire estimée par kata (A) – sélection",
            )
            st.plotly_chart(fig, width="stretch", key="proba_kata_bar_manual")

            st.info(
                "✅ Anti-biais : la proba finale est **shrinkée** vers 50% si A/B ont peu de matchs "
                "et/ou si le kata est peu observé pour A."
            )

        if run_top3:
            a_match_count = int(base_feats["a_matches"])

            if a_match_count >= 4:
                candidates = sorted(set(df_scope[df_scope["Nom"] == nom_a]["Kata"].dropna().astype(str).unique().tolist()))
                source = f"katas déjà joués par A (A a {a_match_count} matchs)"
            else:
                candidates = list(map(str, katas_style))
                source = f"tous les katas du style (A n’a que {a_match_count} matchs)"

            if not candidates:
                st.warning("Impossible de proposer un Top 3 : aucun kata candidat trouvé.")
            else:
                res_df_top = _predict_for_katas(
                    model=model,
                    ag=ag,
                    nom_a=nom_a,
                    nom_b=nom_b,
                    n_tour=str(n_tour),
                    katas=candidates,
                    base_feats=base_feats,
                )

                st.subheader("🏆 Top 3 katas à faire")
                st.caption(f"Source candidats : **{source}**")
                st.dataframe(res_df_top.head(3), width="stretch")

                top10 = res_df_top.head(10).copy()
                fig2 = px.bar(
                    top10,
                    x="Kata",
                    y="Probabilité de victoire (%)",
                    hover_data=["Confiance (0-1)", "Nb occurrences (A, kata)", "Nb matchs (A)", "Nb matchs (B)"],
                    title="Top katas recommandés (Top 10 affiché)",
                )
                st.plotly_chart(fig2, width="stretch", key="proba_kata_bar_top3")
