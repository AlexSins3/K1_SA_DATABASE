# tabs/proba_victoire_kata.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =========================================================
# Utilitaires / normalisations
# =========================================================

def _to_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def _normalize_bool_to_int(v) -> int | None:
    """Retourne 1/0/None (robuste)."""
    if pd.isna(v):
        return None

    if isinstance(v, (bool, np.bool_)):
        return int(bool(v))

    s = str(v).strip().lower()
    if s in {"true", "vrai", "1", "oui", "y", "yes", "win", "gagné", "gagne"}:
        return 1
    if s in {"false", "faux", "0", "non", "n", "no", "lose", "perdu"}:
        return 0

    # fallback
    return int(bool(v))


def _safe_mode(series: pd.Series, default="Non spécifié"):
    m = series.dropna()
    if m.empty:
        return default
    try:
        return m.mode().iloc[0]
    except Exception:
        return m.iloc[0]


def _logit(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    return np.log(p / (1 - p))


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))


def _beta_smooth_rate(wins: float, total: float, alpha: float = 2.0, beta: float = 2.0) -> float:
    """Lissage Beta(a,b): (wins+alpha)/(total+alpha+beta)."""
    return (wins + alpha) / (total + alpha + beta)


def _elo_prob_from_ranking_diff(diff: float) -> float:
    """
    Modèle "prevision victoire elo" (logique Elo) :
    pi = 1 / (1 + 10^(-diff/400))
    """
    try:
        return 1.0 / (1.0 + 10.0 ** (-(diff) / 400.0))
    except Exception:
        return 0.5


# =========================================================
# Construction des matches (paires R/B)
# =========================================================

def _build_paired_matches(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construit un DF match-level en associant les lignes consécutives R/B.
    Hypothèse : la BDD est déjà dans l’ordre où les paires se suivent (comme ton historique).
    """
    d = df.copy().reset_index(drop=True)

    needed = ["Nom", "Ceinture", "Kata", "N_Tour", "Competition", "Type_Compet", "Victoire"]
    for c in needed:
        if c not in d.columns:
            raise ValueError(f"Colonne manquante: {c}")

    # colonnes optionnelles utiles
    if "Year" not in d.columns:
        d["Year"] = np.nan
    if "Ranking" not in d.columns:
        d["Ranking"] = np.nan
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

    # clé “même contexte” pour éviter de pairer des trucs différents
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

            # outcome pour le rouge (1 = rouge gagne)
            y_red = _normalize_bool_to_int(red.get("Victoire"))
            if y_red is None:
                continue

            rows.append(
                {
                    "Competition": red.get("Competition"),
                    "Year": red.get("Year"),
                    "Type_Compet": red.get("Type_Compet"),
                    "N_Tour": red.get("N_Tour"),
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
                    "Red_Ranking": _to_float(red.get("Ranking")),
                    "Blue_Ranking": _to_float(blue.get("Ranking")),
                    "Red_Note": _to_float(red.get("Note")),
                    "Blue_Note": _to_float(blue.get("Note")),
                    "Red_Win": int(y_red),
                }
            )

    return pd.DataFrame(rows)


def _to_directed_rows(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Du match-level => 2 lignes "dirigées" (Athlete vs Opponent).
    Label = Athlete_Win
    """
    if matches.empty:
        return pd.DataFrame()

    rows = []

    for _, m in matches.iterrows():
        # rouge
        rows.append(
            {
                "Nom": m["Red_Nom"],
                "Opponent": m["Blue_Nom"],
                "Kata": m["Red_Kata"],
                "Opp_Kata": m["Blue_Kata"],
                "N_Tour": m["N_Tour"],
                "Competition": m["Competition"],
                "Year": m["Year"],
                "Type_Compet": m["Type_Compet"],
                "Nation": m["Red_Nation"],
                "Opp_Nation": m["Blue_Nation"],
                "Style": m["Red_Style"],
                "Opp_Style": m["Blue_Style"],
                "Sexe": m["Red_Sexe"],
                "Opp_Sexe": m["Blue_Sexe"],
                "Ranking": m["Red_Ranking"],
                "Opp_Ranking": m["Blue_Ranking"],
                "Note": m["Red_Note"],
                "Athlete_Win": int(m["Red_Win"]),
            }
        )
        # bleu
        rows.append(
            {
                "Nom": m["Blue_Nom"],
                "Opponent": m["Red_Nom"],
                "Kata": m["Blue_Kata"],
                "Opp_Kata": m["Red_Kata"],
                "N_Tour": m["N_Tour"],
                "Competition": m["Competition"],
                "Year": m["Year"],
                "Type_Compet": m["Type_Compet"],
                "Nation": m["Blue_Nation"],
                "Opp_Nation": m["Red_Nation"],
                "Style": m["Blue_Style"],
                "Opp_Style": m["Red_Style"],
                "Sexe": m["Blue_Sexe"],
                "Opp_Sexe": m["Red_Sexe"],
                "Ranking": m["Blue_Ranking"],
                "Opp_Ranking": m["Red_Ranking"],
                "Note": m["Blue_Note"],
                "Athlete_Win": int(1 - int(m["Red_Win"])),
            }
        )

    out = pd.DataFrame(rows)
    return out


# =========================================================
# Agrégats & “poids kata” (shrinkage anti-biais)
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

    # overall athlete winrate (smoothed)
    a = d.groupby("Nom")["Athlete_Win"].agg(["sum", "count"]).reset_index()
    a.rename(columns={"sum": "Wins", "count": "Total"}, inplace=True)
    a["WinRate_Smoothed"] = a.apply(lambda r: _beta_smooth_rate(r["Wins"], r["Total"], 2, 2), axis=1)

    # athlete-kata stats
    ak = d.groupby(["Nom", "Kata"])["Athlete_Win"].agg(["sum", "count"]).reset_index()
    ak.rename(columns={"sum": "Kata_Wins", "count": "Kata_Total"}, inplace=True)

    # mean note per athlete-kata
    notes = d.dropna(subset=["Note"]).groupby(["Nom", "Kata"])["Note"].mean().reset_index()
    notes.rename(columns={"Note": "Kata_Note_Mean"}, inplace=True)
    ak = ak.merge(notes, on=["Nom", "Kata"], how="left")

    # smoothed athlete-kata win rate
    ak["Kata_WinRate_Smoothed"] = ak.apply(lambda r: _beta_smooth_rate(r["Kata_Wins"], r["Kata_Total"], 1.5, 1.5), axis=1)

    # opponent weakness vs "kata de l'autre" : pour l'athlète (Nom), pertes quand Opp_Kata = X
    # -> ici on grouppe pertes (1 si perte)
    d["Loss"] = 1 - d["Athlete_Win"]
    ok = d.groupby(["Nom", "Opp_Kata"])["Loss"].agg(["sum", "count"]).reset_index()
    ok.rename(columns={"sum": "Losses_vs_Kata", "count": "Total_vs_Kata"}, inplace=True)
    ok["LossRate_vs_Kata_Smoothed"] = ok.apply(lambda r: _beta_smooth_rate(r["Losses_vs_Kata"], r["Total_vs_Kata"], 1.5, 1.5), axis=1)

    # kata-tour win rate global (smoothed)
    kt = d.groupby(["Kata", "N_Tour"])["Athlete_Win"].agg(["sum", "count"]).reset_index()
    kt.rename(columns={"sum": "Wins", "count": "Total"}, inplace=True)
    kt["Kata_Tour_WinRate_Smoothed"] = kt.apply(lambda r: _beta_smooth_rate(r["Wins"], r["Total"], 2, 2), axis=1)

    # head2head
    # clé triée (A,B) puis on stocke wins de A pour être directionnel après
    tmp = d[["Nom", "Opponent", "Athlete_Win"]].copy()
    tmp["A"] = tmp.apply(lambda r: min(str(r["Nom"]), str(r["Opponent"])), axis=1)
    tmp["B"] = tmp.apply(lambda r: max(str(r["Nom"]), str(r["Opponent"])), axis=1)
    tmp["A_is_Nom"] = tmp["Nom"] == tmp["A"]

    # wins de A (dans (A,B))
    tmp["Win_for_A"] = np.where(tmp["A_is_Nom"], tmp["Athlete_Win"], 1 - tmp["Athlete_Win"])
    h2h = tmp.groupby(["A", "B"])["Win_for_A"].agg(["sum", "count"]).reset_index()
    h2h.rename(columns={"sum": "Wins_A", "count": "Total"}, inplace=True)
    h2h["WinRate_A_Smoothed"] = h2h.apply(lambda r: _beta_smooth_rate(r["Wins_A"], r["Total"], 1.5, 1.5), axis=1)

    # ---------------------------------------------------------
    # Kata effect anti-biais “kata rare joué par des très forts”
    # Idee : on retire l’effet “force globale de l’athlète”
    # residual = (win) - (winrate_athlete_smoothed)
    # puis on moyenne par kata et on shrink vers 0
    # ---------------------------------------------------------
    a_map = a.set_index("Nom")["WinRate_Smoothed"].to_dict()
    d["Athlete_Base"] = d["Nom"].map(a_map).fillna(0.5)
    d["Residual_vs_Base"] = d["Athlete_Win"] - d["Athlete_Base"]

    ke = d.groupby("Kata")["Residual_vs_Base"].agg(["mean", "count"]).reset_index()
    ke.rename(columns={"mean": "Residual_Mean", "count": "Kata_Uses"}, inplace=True)

    # shrinkage : effect = residual_mean * (n / (n + k))
    # k = pseudo-count (plus grand => plus de prudence sur les katas rares)
    k = 25.0
    ke["Kata_Effect"] = ke["Residual_Mean"] * (ke["Kata_Uses"] / (ke["Kata_Uses"] + k))

    return Aggregates(
        athlete_stats=a,
        athlete_kata=ak,
        athlete_oppkata_losses=ok,
        kata_tour=kt,
        h2h=h2h,
        kata_effect=ke,
    )


# =========================================================
# Modèle (logistic regression) basé sur features “numériques”
# =========================================================

FEATURE_COLS = [
    "Elo_Prob",
    "Ranking_Diff",
    "Same_Nation",
    "Same_Style",
    "H2H_Total",
    "H2H_WinRate_A",
    "A_WinRate",
    "B_WinRate",
    "A_Kata_WinRate",
    "A_Kata_Note",
    "B_LossRate_vs_AKata",
    "Kata_Tour_WinRate",
    "Kata_Effect",
    "Is_K1",
]

def _prepare_training_table(directed: pd.DataFrame, ag: Aggregates) -> pd.DataFrame:
    if directed.empty:
        return pd.DataFrame()

    d = directed.copy()

    # ranking diff
    d["Ranking_Diff"] = d["Ranking"].fillna(np.nan) - d["Opp_Ranking"].fillna(np.nan)
    # fill ranking diff missing => 0 (neutral)
    d["Ranking_Diff"] = d["Ranking_Diff"].fillna(0.0)

    # Elo prob from ranking diff
    d["Elo_Prob"] = d["Ranking_Diff"].apply(_elo_prob_from_ranking_diff)

    # Same nation / style
    d["Same_Nation"] = (d["Nation"].astype(str) == d["Opp_Nation"].astype(str)).astype(int)
    d["Same_Style"] = (d["Style"].astype(str) == d["Opp_Style"].astype(str)).astype(int)

    # type compet
    d["Is_K1"] = (d["Type_Compet"].astype(str) == "K1").astype(int)

    # overall win rates
    a_map = ag.athlete_stats.set_index("Nom")["WinRate_Smoothed"].to_dict()
    d["A_WinRate"] = d["Nom"].map(a_map).fillna(0.5)
    d["B_WinRate"] = d["Opponent"].map(a_map).fillna(0.5)

    # athlete-kata rate + note
    ak = ag.athlete_kata.set_index(["Nom", "Kata"])
    d["A_Kata_WinRate"] = d.apply(
        lambda r: float(ak.loc[(r["Nom"], r["Kata"]), "Kata_WinRate_Smoothed"])
        if (r["Nom"], r["Kata"]) in ak.index else 0.5,
        axis=1,
    )
    d["A_Kata_Note"] = d.apply(
        lambda r: float(ak.loc[(r["Nom"], r["Kata"]), "Kata_Note_Mean"])
        if (r["Nom"], r["Kata"]) in ak.index and pd.notna(ak.loc[(r["Nom"], r["Kata"]), "Kata_Note_Mean"]) else 0.0,
        axis=1,
    )

    # opp loss vs A kata
    ok = ag.athlete_oppkata_losses.set_index(["Nom", "Opp_Kata"])
    d["B_LossRate_vs_AKata"] = d.apply(
        lambda r: float(ok.loc[(r["Opponent"], r["Kata"]), "LossRate_vs_Kata_Smoothed"])
        if (r["Opponent"], r["Kata"]) in ok.index else 0.5,
        axis=1,
    )

    # kata-tour
    kt = ag.kata_tour.set_index(["Kata", "N_Tour"])
    d["Kata_Tour_WinRate"] = d.apply(
        lambda r: float(kt.loc[(r["Kata"], r["N_Tour"]), "Kata_Tour_WinRate_Smoothed"])
        if (r["Kata"], r["N_Tour"]) in kt.index else 0.5,
        axis=1,
    )

    # kata effect anti-biais
    ke = ag.kata_effect.set_index("Kata")["Kata_Effect"].to_dict()
    d["Kata_Effect"] = d["Kata"].map(ke).fillna(0.0)

    # head2head
    h = ag.h2h.copy()
    if not h.empty:
        h_index = h.set_index(["A", "B"])

        def get_h2h(nom: str, opp: str) -> Tuple[float, float]:
            a = min(str(nom), str(opp))
            b = max(str(nom), str(opp))
            if (a, b) not in h_index.index:
                return 0.0, 0.5
            row = h_index.loc[(a, b)]
            total = float(row["Total"])
            # winrate pour A (dans le couple (A,B))
            winrate_A = float(row["WinRate_A_Smoothed"])
            # si nom == A => winrate nom = winrate_A, sinon = 1-winrate_A
            wr = winrate_A if str(nom) == a else (1.0 - winrate_A)
            return total, wr

        h2h_vals = d.apply(lambda r: get_h2h(r["Nom"], r["Opponent"]), axis=1)
        d["H2H_Total"] = [v[0] for v in h2h_vals]
        d["H2H_WinRate_A"] = [v[1] for v in h2h_vals]
    else:
        d["H2H_Total"] = 0.0
        d["H2H_WinRate_A"] = 0.5

    # target
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

    if train.empty:
        # modèle “neutre”
        pipe = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=200, C=1.0, solver="lbfgs")),
            ]
        )
        # fit fake
        X = np.zeros((10, len(FEATURE_COLS)))
        y = np.array([0, 1] * 5)
        pipe.fit(X, y)
        return pipe, ag

    X = train[FEATURE_COLS].values
    y = train["y"].values

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, C=1.0, solver="lbfgs")),
        ]
    )
    pipe.fit(X, y)
    return pipe, ag


def _confidence_score(ag: Aggregates, nom: str, kata: str) -> float:
    """
    Score 0..1 basé sur la quantité d’info :
    - nb fois où A a fait ce kata
    - nb d’occurrences globales du kata (via Kata_Effect)
    """
    a_uses = 0
    if not ag.athlete_kata.empty:
        sub = ag.athlete_kata[(ag.athlete_kata["Nom"] == nom) & (ag.athlete_kata["Kata"] == kata)]
        if not sub.empty:
            a_uses = int(sub["Kata_Total"].iloc[0])

    global_uses = 0
    if not ag.kata_effect.empty:
        sub2 = ag.kata_effect[ag.kata_effect["Kata"] == kata]
        if not sub2.empty:
            global_uses = int(sub2["Kata_Uses"].iloc[0])

    # mapping simple (saturant)
    s1 = 1.0 - np.exp(-a_uses / 6.0)       # A a déjà de l'historique sur le kata
    s2 = 1.0 - np.exp(-global_uses / 25.0) # kata globalement observé
    return float(0.55 * s1 + 0.45 * s2)


# =========================================================
# Onglet Streamlit
# =========================================================

def show_proba_victoire_kata_tab(data: pd.DataFrame) -> None:
    st.header("Probabilité de victoire par kata")

    st.markdown(
        """
**Avertissement**  
- Les résultats sont des **probabilités**, pas une vérité absolue.  
- Les katas rares (ex: Enpi) sont **corrigés** via un système de *poids / shrinkage* :
  si un kata est très peu observé et joué surtout par des athlètes très forts, on évite de lui attribuer un avantage artificiel.
        """
    )

    df = data.copy()

    # normalisations légères (au cas où)
    if "Note" in df.columns and df["Note"].dtype == object:
        df["Note"] = df["Note"].astype(str).str.replace(",", ".", regex=False)
    for col in ["Ranking", "Age", "Year", "Note"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Train model + aggs (cache)
    model, ag = _train_model_and_aggs(df)

    # =========================================================
    # Bande filtres
    # =========================================================
    filters_col, content_col = st.columns([0.9, 2.4])

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
        selected_type_compet = st.radio(
            "Type de compétition",
            type_compet_options,
            key="proba_type_compet",
        )

        df_scope = df.copy()
        if selected_type_compet == "Premier League (K1)":
            df_scope = df_scope[df_scope["Type_Compet"] == "K1"]
        elif selected_type_compet == "Series A (SA)":
            df_scope = df_scope[df_scope["Type_Compet"] == "SA"]

        # liste athlètes
        athlete_names = sorted(df_scope["Nom"].dropna().unique().tolist())
        if not athlete_names:
            st.warning("Aucun athlète disponible avec ces filtres.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        nom_a = st.selectbox("Athlète A", athlete_names, key="proba_nom_a")

        # sexe de A -> filtre pour B
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

        # Tour
        tour_options = sorted(df_scope["N_Tour"].dropna().astype(str).unique().tolist())
        if not tour_options:
            st.warning("Aucun tour (N_Tour) disponible dans ce périmètre.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        n_tour = st.selectbox("Tour (N_Tour)", tour_options, key="proba_tour")

        # Styles
        style_a = _safe_mode(df_scope[df_scope["Nom"] == nom_a]["Style"], default=None)
        style_b = _safe_mode(df_scope[df_scope["Nom"] == nom_b]["Style"], default=None)

        st.markdown("---")
        st.markdown("#### 🥋 Katas testés (pour A)")

        # katas disponibles : on privilégie ceux vus dans le scope + (optionnel) filtrage par style
        katas_scope = df_scope["Kata"].dropna().astype(str).unique().tolist()

        if style_a is not None and "Style" in df_scope.columns:
            # On garde les katas qui apparaissent avec ce style
            tmp = df_scope[["Kata", "Style"]].dropna()
            katas_style = sorted(tmp[tmp["Style"].astype(str) == str(style_a)]["Kata"].astype(str).unique().tolist())

            # cas spécial Suparinpei : on le laisse passer si présent
            if "Suparinpei" in katas_scope and "Suparinpei" not in katas_style:
                katas_style.append("Suparinpei")
                katas_style = sorted(set(katas_style))
        else:
            katas_style = sorted(set(map(str, katas_scope)))

        # default = katas déjà faits par A (dans ce scope)
        katas_effectues_a = sorted(
            df_scope[df_scope["Nom"] == nom_a]["Kata"].dropna().astype(str).unique().tolist()
        )

        katas_selectionnes = st.multiselect(
            "Katas à tester",
            options=katas_style,
            default=katas_effectues_a if katas_effectues_a else katas_style,
            key="proba_katas",
        )

        st.markdown("</div>", unsafe_allow_html=True)

    # =========================================================
    # Contenu / calcul
    # =========================================================
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

        if not katas_selectionnes:
            st.info("Sélectionne au moins un kata à tester.")
            return

        if st.button("Calculer les probabilités", key="proba_run"):
            # infos de base athlètes
            ranking_a = df_scope[df_scope["Nom"] == nom_a]["Ranking"].dropna().mean()
            ranking_b = df_scope[df_scope["Nom"] == nom_b]["Ranking"].dropna().mean()

            # si ranking absent => neutralité (diff=0) plutôt que hack “max+1”
            if np.isnan(ranking_a):
                ranking_a = np.nan
            if np.isnan(ranking_b):
                ranking_b = np.nan

            ranking_diff = float((ranking_a - ranking_b)) if (pd.notna(ranking_a) and pd.notna(ranking_b)) else 0.0
            elo_prob = _elo_prob_from_ranking_diff(ranking_diff)

            nation_a = _safe_mode(df_scope[df_scope["Nom"] == nom_a]["Nation"], default="")
            nation_b = _safe_mode(df_scope[df_scope["Nom"] == nom_b]["Nation"], default="")
            same_nation = int(str(nation_a) == str(nation_b))

            same_style = int(str(style_a) == str(style_b))

            # h2h
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

            # overall win rates
            a_map = {}
            if not ag.athlete_stats.empty:
                a_map = ag.athlete_stats.set_index("Nom")["WinRate_Smoothed"].to_dict()
            a_win = float(a_map.get(nom_a, 0.5))
            b_win = float(a_map.get(nom_b, 0.5))

            # opp weakness table
            ok_index = None
            if not ag.athlete_oppkata_losses.empty:
                ok_index = ag.athlete_oppkata_losses.set_index(["Nom", "Opp_Kata"])

            # athlete-kata table
            ak_index = None
            if not ag.athlete_kata.empty:
                ak_index = ag.athlete_kata.set_index(["Nom", "Kata"])

            # kata-tour table
            kt_index = None
            if not ag.kata_tour.empty:
                kt_index = ag.kata_tour.set_index(["Kata", "N_Tour"])

            # kata-effect map
            ke_map = {}
            if not ag.kata_effect.empty:
                ke_map = ag.kata_effect.set_index("Kata")["Kata_Effect"].to_dict()

            is_k1 = 1 if selected_type_compet == "Premier League (K1)" else (0 if selected_type_compet == "Series A (SA)" else 0)
            # si "Tous", on met 0 : le modèle le gère via autres features

            results = []
            for kata in katas_selectionnes:
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
                        elo_prob,
                        ranking_diff,
                        same_nation,
                        same_style,
                        h2h_total,
                        h2h_wr_a,
                        a_win,
                        b_win,
                        a_kata_wr,
                        a_kata_note,
                        b_loss_vs,
                        kata_tour_wr,
                        kata_effect,
                        is_k1,
                    ]],
                    dtype=float,
                )

                proba = float(model.predict_proba(X)[0, 1]) * 100.0
                conf = _confidence_score(ag, nom_a, kata)  # 0..1

                results.append(
                    {
                        "Kata": kata,
                        "Probabilité de victoire (%)": round(proba, 3),
                        "Confiance (0-1)": round(conf, 3),
                        "Nb occurrences (A, kata)": int(a_kata_n),
                    }
                )

            res_df = pd.DataFrame(results).sort_values("Probabilité de victoire (%)", ascending=False).reset_index(drop=True)

            st.subheader(f"Résultats – {nom_a} vs {nom_b} (tour: {n_tour})")

            # Top 3
            st.markdown("##### Top 3")
            st.dataframe(res_df.head(3), width="stretch")

            st.markdown("##### Détail complet")
            st.dataframe(res_df, width="stretch")

            # Bar chart
            fig = px.bar(
                res_df,
                x="Kata",
                y="Probabilité de victoire (%)",
                hover_data=["Confiance (0-1)", "Nb occurrences (A, kata)"],
                title="Probabilité de victoire estimée par kata (A)",
            )
            st.plotly_chart(fig, width="stretch", key="proba_kata_bar")

            st.info(
                "💡 **Lecture de la “Confiance”** : plus elle est proche de 1, plus il y a d’historique "
                "sur ce kata (chez A + global). Les katas rares sont volontairement pénalisés/ramenés vers la moyenne."
            )
