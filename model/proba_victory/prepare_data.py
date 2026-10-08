"""
Préparation des données V2 — Modèle de prédiction de victoire aux drapeaux.
============================================================================

Améliorations vs V1 :
- Features riches inspirées du modèle kata (kata WR, style, H2H, diversité, trend...)
- Transfert learning : modèle de dominance pré-entraîné sur les notes 2024-2025
- Modèles séparés K1 (7 juges) et SA (5 juges)
- Pipeline anti-leakage (tout calculé de façon cumulative/avant le match)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ──
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # model/proba_victory/ → model/ → K1_SA_DATABASE/

# ── Constantes chronologiques ──
COMPET_CHRONO_ORDER = {
    ("K1 Paris", 2024): 1, ("K1 Antalya", 2024): 2, ("K1 Cairo", 2024): 3,
    ("K1 Casablanca", 2024): 4, ("SA Athens", 2024): 5, ("SA Larnaca", 2024): 6,
    ("SA Salzbourg", 2024): 7,
    ("K1 Hanghou", 2025): 8, ("K1 Rabat", 2025): 9, ("SA KualaLumpur", 2025): 10,
    ("SA Kuala Lumpur", 2025): 10, ("K1 Cairo", 2025): 11, ("K1 Paris", 2025): 12,
    ("SA Larnaca", 2025): 13, ("SA Salzbourg", 2025): 14, ("SA Tbilisi", 2025): 15,
    ("K1 Istanbul", 2026): 16, ("SA Tbilisi", 2026): 17, ("K1 Roma", 2026): 18,
    ("K1 Leshan", 2026): 19, ("SA ACoruna", 2026): 20, ("SA A Coruna", 2026): 20,
    ("K1 Rabat", 2026): 21, ("SA Salzbourg", 2026): 22,
}


def get_compet_chrono_rank(competition: str, year=None) -> int:
    """Return chronological rank of a competition (higher = more recent)."""
    comp_str = str(competition).replace("_", " ") if competition else ""
    year_int = None
    try:
        year_int = int(year) if year is not None else None
    except (ValueError, TypeError):
        pass
    if year_int is not None:
        rank = COMPET_CHRONO_ORDER.get((comp_str, year_int), 0)
        if rank > 0:
            return rank
        year_ranks = [r for (_, y), r in COMPET_CHRONO_ORDER.items() if y == year_int]
        if year_ranks:
            return max(year_ranks)
        max_known_rank = max(COMPET_CHRONO_ORDER.values()) if COMPET_CHRONO_ORDER else 0
        max_known_year = max(y for (_, y) in COMPET_CHRONO_ORDER.keys()) if COMPET_CHRONO_ORDER else 2024
        return max_known_rank + (year_int - max_known_year) * 7
    return 0


# ══════════════════════════════════════════════════════════════════════════════
# 1. CHARGEMENT & APPAIRAGE
# ══════════════════════════════════════════════════════════════════════════════

def load_raw_data() -> pd.DataFrame:
    """Charge le CSV brut."""
    csv_path = PROJECT_ROOT / "data" / "Database_K1_SA.csv"
    df = pd.read_csv(csv_path, sep=";")
    df["Note"] = pd.to_numeric(
        df["Note"].astype(str).str.replace(",", ".", regex=False), errors="coerce"
    )
    df["Drapeau"] = pd.to_numeric(df["Drapeau"], errors="coerce")
    df["Ranking"] = pd.to_numeric(df["Ranking"], errors="coerce")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Victoire_bool"] = df["Victoire"].astype(str).str.lower().isin(
        ["true", "1", "vrai", "yes"]
    )
    return df


def pair_matches(df: pd.DataFrame) -> pd.DataFrame:
    """
    Appaire les lignes R/B consécutives pour créer des matchs.
    Validation : un gagnant et un perdant dans chaque paire.
    """
    d = df.copy()
    d = d[d["Nom"].notna() & (d["Nom"].astype(str).str.strip() != "")].reset_index(drop=True)

    n = len(d)
    pairs = []

    i = 0
    while i < n - 1:
        row1 = d.iloc[i]
        row2 = d.iloc[i + 1]

        b1 = str(row1["Ceinture"])
        b2 = str(row2["Ceinture"])

        is_pair = (
            b1 in ("R", "B") and b2 in ("R", "B")
            and b1 != b2
            and row1["Competition"] == row2["Competition"]
            and row1["N_Tour"] == row2["N_Tour"]
            and row1["Year"] == row2["Year"]
            and row1["Type_Compet"] == row2["Type_Compet"]
            and row1["Victoire_bool"] != row2["Victoire_bool"]
        )

        if is_pair:
            pairs.append((i, i + 1))
            i += 2
        else:
            i += 1

    if not pairs:
        return pd.DataFrame()

    idx1 = [p[0] for p in pairs]
    idx2 = [p[1] for p in pairs]

    r1 = d.iloc[idx1].reset_index(drop=True)
    r2 = d.iloc[idx2].reset_index(drop=True)

    # Red = A, Blue = B
    is_r1_red = r1["Ceinture"].astype(str).values == "R"

    def pick_red(col):
        return np.where(is_r1_red, r1[col].values, r2[col].values)

    def pick_blue(col):
        return np.where(is_r1_red, r2[col].values, r1[col].values)

    v1 = r1["Victoire_bool"].values.astype(int)
    v2 = r2["Victoire_bool"].values.astype(int)
    red_win = np.where(is_r1_red, v1, v2)

    matches = pd.DataFrame({
        "Competition": pick_red("Competition"),
        "Year": pick_red("Year").astype(float),
        "Type_Compet": pick_red("Type_Compet"),
        "N_Tour": pick_red("N_Tour"),
        "Sexe": pick_red("Sexe"),
        # Athlète A (Red)
        "A_Nom": pick_red("Nom"),
        "A_Kata": pick_red("Kata"),
        "A_Style": pick_red("Style"),
        "A_Nation": pick_red("Nation"),
        "A_Continent": pick_red("Continent"),
        "A_Region": pick_red("Region_monde"),
        "A_Ranking": pick_red("Ranking").astype(float),
        "A_Age": pick_red("Age").astype(float),
        "A_Note": pick_red("Note").astype(float),
        "A_Drapeau": pick_red("Drapeau").astype(float),
        # Athlète B (Blue)
        "B_Nom": pick_blue("Nom"),
        "B_Kata": pick_blue("Kata"),
        "B_Style": pick_blue("Style"),
        "B_Nation": pick_blue("Nation"),
        "B_Continent": pick_blue("Continent"),
        "B_Region": pick_blue("Region_monde"),
        "B_Ranking": pick_blue("Ranking").astype(float),
        "B_Age": pick_blue("Age").astype(float),
        "B_Note": pick_blue("Note").astype(float),
        "B_Drapeau": pick_blue("Drapeau").astype(float),
        # Target
        "A_Win": red_win,
    })

    # Chrono rank
    matches["chrono_rank"] = matches.apply(
        lambda r: get_compet_chrono_rank(str(r["Competition"]).replace("_", " "), r["Year"]),
        axis=1
    )
    matches = matches.sort_values("chrono_rank").reset_index(drop=True)

    return matches


# ══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING — COMPLET
# ══════════════════════════════════════════════════════════════════════════════

def engineer_features(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering complet avec calcul cumulatif (anti-leakage).
    Utilise TOUT l'historique (notes + drapeaux) pour construire les features.
    """
    df = matches.copy()

    # ──────────────────────────────────────────────────────────────────────
    # FEATURES STATIQUES (directement calculables sans historique)
    # ──────────────────────────────────────────────────────────────────────

    # Différentiels de base
    df["Diff_Ranking"] = df["B_Ranking"] - df["A_Ranking"]  # Positif = A mieux classé
    df["Log_Ranking_Ratio"] = np.log1p(df["B_Ranking"]) - np.log1p(df["A_Ranking"])
    df["Diff_Age"] = (df["A_Age"] - df["B_Age"]) / 10.0  # Normalisé

    # Indicatrices contextuelles
    df["Same_Nation"] = (df["A_Nation"] == df["B_Nation"]).astype(int)
    df["Same_Continent"] = (df["A_Continent"] == df["B_Continent"]).astype(int)
    df["Same_Style"] = (df["A_Style"] == df["B_Style"]).astype(int)
    df["Is_K1"] = (df["Type_Compet"] == "K1").astype(int)
    df["Is_Male"] = (df["Sexe"] == "M").astype(int)
    df["Max_Flags"] = df["Is_K1"].map({1: 7, 0: 5})

    # Tour rank (ordinal)
    tour_order = {
        "Pool_1": 1, "Pool_2": 2, "Pool_3": 3,
        "T1": 1, "T2": 2, "T3": 3,
        "PW1": 4, "PW2": 5, "PW3": 6,
        "RP1": 4, "RP2": 5, "RP3": 6, "RP4": 7,
        "R1": 7, "R2": 8,
        "Bronze": 9, "Final": 10,
    }
    df["Tour_Rank"] = df["N_Tour"].map(tour_order).fillna(5) / 10.0

    # Is_Home : A joue dans son continent
    df["Is_Home_A"] = (df["A_Continent"].astype(str) == df["A_Region"].astype(str)).astype(int)
    df["Is_Home_B"] = (df["B_Continent"].astype(str) == df["B_Region"].astype(str)).astype(int)
    df["Home_Advantage"] = df["Is_Home_A"] - df["Is_Home_B"]

    # ──────────────────────────────────────────────────────────────────────
    # FEATURES HISTORIQUES (cumulatif, calculé AVANT chaque match)
    # ──────────────────────────────────────────────────────────────────────

    # Structures cumulatives
    athlete_wins = {}      # nom → [wins, total]
    athlete_notes = {}     # nom → list of notes
    athlete_flags = {}     # nom → list of flags obtained
    athlete_recent = {}    # nom → list of (win: 0/1) pour les derniers matchs
    athlete_kata_wins = {} # (nom, kata) → [wins, total]
    athlete_kata_notes = {}  # (nom, kata) → list of notes
    h2h_record = {}        # frozenset(a,b) → {a: wins_a, b: wins_b}
    kata_wins = {}         # kata → [wins, total]
    athlete_katas_used = {}  # nom → set of katas

    # Résultats
    feats = {
        "A_Winrate": [], "B_Winrate": [],
        "A_Recent_Form": [], "B_Recent_Form": [],
        "A_H2H_Rate": [],
        "A_Kata_WR": [], "B_Kata_WR": [],
        "A_Note_Mean": [], "B_Note_Mean": [],
        "A_Kata_Note_Mean": [],
        "A_Nb_Matchs": [], "B_Nb_Matchs": [],
        "A_Kata_Diversity": [], "B_Kata_Diversity": [],
        "A_Note_Std": [], "B_Note_Std": [],
        "A_Flag_Mean": [], "B_Flag_Mean": [],
        "Kata_A_Global_WR": [], "Kata_B_Global_WR": [],
        "A_Note_Trend": [], "B_Note_Trend": [],
    }

    window_recent = 10

    for _, row in df.iterrows():
        a, b = row["A_Nom"], row["B_Nom"]
        a_kata, b_kata = row["A_Kata"], row["B_Kata"]

        # ── Récupération des stats AVANT ce match ──
        a_stats = athlete_wins.get(a, [0, 0])
        b_stats = athlete_wins.get(b, [0, 0])

        # Winrate
        feats["A_Winrate"].append(a_stats[0] / a_stats[1] if a_stats[1] > 0 else 0.5)
        feats["B_Winrate"].append(b_stats[0] / b_stats[1] if b_stats[1] > 0 else 0.5)

        # Nb matchs (expérience)
        feats["A_Nb_Matchs"].append(a_stats[1])
        feats["B_Nb_Matchs"].append(b_stats[1])

        # Forme récente
        a_rec = athlete_recent.get(a, [])
        b_rec = athlete_recent.get(b, [])
        feats["A_Recent_Form"].append(np.mean(a_rec[-window_recent:]) if a_rec else 0.5)
        feats["B_Recent_Form"].append(np.mean(b_rec[-window_recent:]) if b_rec else 0.5)

        # H2H
        key = frozenset([a, b])
        if key in h2h_record:
            rec = h2h_record[key]
            total_h2h = sum(rec.values())
            feats["A_H2H_Rate"].append(rec.get(a, 0) / total_h2h if total_h2h > 0 else 0.5)
        else:
            feats["A_H2H_Rate"].append(0.5)

        # Kata WR (A avec son kata, B avec son kata)
        ak_stats = athlete_kata_wins.get((a, a_kata), [0, 0])
        bk_stats = athlete_kata_wins.get((b, b_kata), [0, 0])
        feats["A_Kata_WR"].append((ak_stats[0] + 1) / (ak_stats[1] + 2) if ak_stats[1] > 0 else 0.5)
        feats["B_Kata_WR"].append((bk_stats[0] + 1) / (bk_stats[1] + 2) if bk_stats[1] > 0 else 0.5)

        # Note moyenne globale par athlète
        a_notes_list = athlete_notes.get(a, [])
        b_notes_list = athlete_notes.get(b, [])
        feats["A_Note_Mean"].append(np.mean(a_notes_list) if a_notes_list else 39.0)
        feats["B_Note_Mean"].append(np.mean(b_notes_list) if b_notes_list else 39.0)

        # Note std (régularité)
        feats["A_Note_Std"].append(np.std(a_notes_list) if len(a_notes_list) >= 3 else 2.0)
        feats["B_Note_Std"].append(np.std(b_notes_list) if len(b_notes_list) >= 3 else 2.0)

        # Note trend (pente simple : moyenne 2nde moitié - 1ère moitié)
        def _note_trend(notes_list):
            if len(notes_list) < 4:
                return 0.0
            mid = len(notes_list) // 2
            return np.mean(notes_list[mid:]) - np.mean(notes_list[:mid])

        feats["A_Note_Trend"].append(_note_trend(a_notes_list))
        feats["B_Note_Trend"].append(_note_trend(b_notes_list))

        # Note moyenne par kata (A uniquement, pertinence du choix)
        ak_notes = athlete_kata_notes.get((a, a_kata), [])
        feats["A_Kata_Note_Mean"].append(np.mean(ak_notes) if ak_notes else 39.0)

        # Kata diversity
        a_katas = athlete_katas_used.get(a, set())
        b_katas = athlete_katas_used.get(b, set())
        feats["A_Kata_Diversity"].append(len(a_katas) / max(a_stats[1], 1))
        feats["B_Kata_Diversity"].append(len(b_katas) / max(b_stats[1], 1))

        # Flag mean (historique drapeaux)
        a_flags = athlete_flags.get(a, [])
        b_flags = athlete_flags.get(b, [])
        feats["A_Flag_Mean"].append(np.mean(a_flags) if a_flags else 2.5)
        feats["B_Flag_Mean"].append(np.mean(b_flags) if b_flags else 2.5)

        # Kata global WR (popularité/force du kata globalement)
        ka_stats = kata_wins.get(a_kata, [0, 0])
        kb_stats = kata_wins.get(b_kata, [0, 0])
        feats["Kata_A_Global_WR"].append((ka_stats[0] + 2) / (ka_stats[1] + 4) if ka_stats[1] > 0 else 0.5)
        feats["Kata_B_Global_WR"].append((kb_stats[0] + 2) / (kb_stats[1] + 4) if kb_stats[1] > 0 else 0.5)

        # ── Mise à jour APRÈS ce match ──
        win_a = int(row["A_Win"])

        # Wins/Total
        if a not in athlete_wins:
            athlete_wins[a] = [0, 0]
        if b not in athlete_wins:
            athlete_wins[b] = [0, 0]
        athlete_wins[a][1] += 1
        athlete_wins[b][1] += 1
        athlete_wins[a][0] += win_a
        athlete_wins[b][0] += (1 - win_a)

        # Recent
        if a not in athlete_recent:
            athlete_recent[a] = []
        if b not in athlete_recent:
            athlete_recent[b] = []
        athlete_recent[a].append(win_a)
        athlete_recent[b].append(1 - win_a)

        # H2H
        if key not in h2h_record:
            h2h_record[key] = {}
        winner = a if win_a else b
        h2h_record[key][winner] = h2h_record[key].get(winner, 0) + 1

        # Kata wins
        if (a, a_kata) not in athlete_kata_wins:
            athlete_kata_wins[(a, a_kata)] = [0, 0]
        if (b, b_kata) not in athlete_kata_wins:
            athlete_kata_wins[(b, b_kata)] = [0, 0]
        athlete_kata_wins[(a, a_kata)][1] += 1
        athlete_kata_wins[(b, b_kata)][1] += 1
        athlete_kata_wins[(a, a_kata)][0] += win_a
        athlete_kata_wins[(b, b_kata)][0] += (1 - win_a)

        # Notes
        if not np.isnan(row["A_Note"]):
            if a not in athlete_notes:
                athlete_notes[a] = []
            athlete_notes[a].append(row["A_Note"])
            if (a, a_kata) not in athlete_kata_notes:
                athlete_kata_notes[(a, a_kata)] = []
            athlete_kata_notes[(a, a_kata)].append(row["A_Note"])
        if not np.isnan(row["B_Note"]):
            if b not in athlete_notes:
                athlete_notes[b] = []
            athlete_notes[b].append(row["B_Note"])
            if (b, b_kata) not in athlete_kata_notes:
                athlete_kata_notes[(b, b_kata)] = []
            athlete_kata_notes[(b, b_kata)].append(row["B_Note"])

        # Flags
        if not np.isnan(row.get("A_Drapeau", np.nan)):
            if a not in athlete_flags:
                athlete_flags[a] = []
            athlete_flags[a].append(row["A_Drapeau"])
        if not np.isnan(row.get("B_Drapeau", np.nan)):
            if b not in athlete_flags:
                athlete_flags[b] = []
            athlete_flags[b].append(row["B_Drapeau"])

        # Kata diversity
        if a not in athlete_katas_used:
            athlete_katas_used[a] = set()
        if b not in athlete_katas_used:
            athlete_katas_used[b] = set()
        athlete_katas_used[a].add(a_kata)
        athlete_katas_used[b].add(b_kata)

        # Kata global
        if a_kata not in kata_wins:
            kata_wins[a_kata] = [0, 0]
        if b_kata not in kata_wins:
            kata_wins[b_kata] = [0, 0]
        kata_wins[a_kata][1] += 1
        kata_wins[b_kata][1] += 1
        kata_wins[a_kata][0] += win_a
        kata_wins[b_kata][0] += (1 - win_a)

    # Assigner les features
    for col, values in feats.items():
        df[col] = values

    # ── Features dérivées ──
    df["Diff_Winrate"] = df["A_Winrate"] - df["B_Winrate"]
    df["Diff_Form"] = df["A_Recent_Form"] - df["B_Recent_Form"]
    df["Diff_Note_Mean"] = df["A_Note_Mean"] - df["B_Note_Mean"]
    df["Diff_Kata_WR"] = df["A_Kata_WR"] - df["B_Kata_WR"]
    df["Diff_Kata_Diversity"] = df["A_Kata_Diversity"] - df["B_Kata_Diversity"]
    df["Diff_Note_Std"] = df["A_Note_Std"] - df["B_Note_Std"]  # Négatif = A plus régulier
    df["Diff_Flag_Mean"] = df["A_Flag_Mean"] - df["B_Flag_Mean"]
    df["Diff_Note_Trend"] = df["A_Note_Trend"] - df["B_Note_Trend"]
    df["Exp_Diff"] = np.log1p(df["A_Nb_Matchs"]) - np.log1p(df["B_Nb_Matchs"])
    df["Diff_Kata_Global_WR"] = df["Kata_A_Global_WR"] - df["Kata_B_Global_WR"]

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3. TRANSFERT LEARNING + FEATURES INTERACTIVES
# ══════════════════════════════════════════════════════════════════════════════

def add_transfer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute :
    1. Le Transfer_Score du vrai modèle de transfert (entraîné sur notes 2024-2025)
    2. Les features interactives (Sexe × ranking, Tour × ranking, etc.)
    """
    df = df.copy()

    # ── 1. Vrai modèle de transfert ──
    # On essaie de charger le modèle de transfert. S'il n'existe pas, on utilise le proxy.
    try:
        from transfer_model import predict_dominance, compute_transfer_features, TRANSFER_FEATURES
        
        # Le modèle de transfert a besoin de ses propres features
        # On les reconstruit ici (certaines sont déjà calculées)
        # On crée un DataFrame temporaire avec les colonnes attendues
        tf_df = df.copy()
        # Ajouter les colonnes manquantes pour le transfert
        if "Is_Male" not in tf_df.columns:
            tf_df["Is_Male"] = (tf_df["Sexe"] == "M").astype(int)
        if "Ranking_x_Tour" not in tf_df.columns:
            tf_df["Ranking_x_Tour"] = tf_df["Log_Ranking_Ratio"] * tf_df["Tour_Rank"]
        if "Ranking_x_IsMale" not in tf_df.columns:
            tf_df["Ranking_x_IsMale"] = tf_df["Log_Ranking_Ratio"] * tf_df["Is_Male"]
        
        # Vérifier que toutes les features du transfert sont présentes
        missing = [f for f in TRANSFER_FEATURES if f not in tf_df.columns]
        if not missing:
            df["Transfer_Score"] = predict_dominance(tf_df)
            print("   ✓ Transfer_Score calculé via le modèle de transfert")
        else:
            # Fallback: proxy linéaire
            df["Transfer_Score"] = (
                0.4 * df["Diff_Winrate"]
                + 0.3 * df["Log_Ranking_Ratio"]
                + 0.2 * df.get("Diff_Note_Mean", 0) / 2.0
                + 0.1 * df.get("Diff_Form", 0)
            )
            print(f"   ⚠️ Transfer fallback (features manquantes: {missing})")
    except (ImportError, FileNotFoundError, Exception) as e:
        # Fallback: proxy linéaire
        df["Transfer_Score"] = (
            0.4 * df["Diff_Winrate"]
            + 0.3 * df["Log_Ranking_Ratio"]
            + 0.2 * df.get("Diff_Note_Mean", 0) / 2.0
            + 0.1 * df.get("Diff_Form", 0)
        )
        print(f"   ⚠️ Transfer fallback ({e})")

    # ── 2. Features interactives ──
    # Interactions Sexe (les femmes et hommes ont des patterns de score différents)
    df["Ranking_x_IsMale"] = df["Log_Ranking_Ratio"] * df["Is_Male"]
    df["Winrate_x_IsMale"] = df["Diff_Winrate"] * df["Is_Male"]

    # Interactions Tour (en finale les scores sont plus serrés)
    df["Ranking_x_Tour"] = df["Log_Ranking_Ratio"] * df["Tour_Rank"]
    df["Winrate_x_Tour"] = df["Diff_Winrate"] * df["Tour_Rank"]
    df["Form_x_Tour"] = df["Diff_Form"] * df["Tour_Rank"]

    # Kata note centrée (pertinence du choix de kata A vs niveau B)
    df["A_Kata_Note_Centered"] = df["A_Kata_Note_Mean"] - df["B_Note_Mean"]

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 4. CONSTRUCTION DE LA TARGET
# ══════════════════════════════════════════════════════════════════════════════

def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Target : nombre de drapeaux de A.
    Filtrage : matchs avec drapeaux valides uniquement.
    """
    df = df.copy()
    df_flags = df[df["A_Drapeau"].notna() & df["B_Drapeau"].notna()].copy()
    df_flags["Target"] = df_flags["A_Drapeau"].astype(int)

    # Vérification cohérence
    total_flags = df_flags["A_Drapeau"] + df_flags["B_Drapeau"]
    expected = df_flags["Max_Flags"]
    inconsistent = (total_flags != expected).sum()
    if inconsistent > 0:
        print(f"⚠️  {inconsistent} matchs avec total drapeaux incohérent (exclus)")
        df_flags = df_flags[total_flags == expected]

    return df_flags


# ══════════════════════════════════════════════════════════════════════════════
# 5. SPLIT CHRONOLOGIQUE
# ══════════════════════════════════════════════════════════════════════════════

def split_chronological(df: pd.DataFrame, test_ratio: float = 0.2) -> tuple:
    """Split train/test chronologique."""
    df = df.sort_values("chrono_rank").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_ratio))
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    return train, test


# ══════════════════════════════════════════════════════════════════════════════
# 6. DÉFINITION DES FEATURES
# ══════════════════════════════════════════════════════════════════════════════

FEATURE_COLS = [
    # Différentiels de ranking/expérience
    "Diff_Ranking", "Log_Ranking_Ratio", "Exp_Diff",
    # Différentiels de performance
    "Diff_Winrate", "Diff_Form", "Diff_Note_Mean", "Diff_Note_Trend",
    # Kata-specific
    "Diff_Kata_WR", "Diff_Kata_Global_WR", "Diff_Kata_Diversity",
    "A_Kata_Note_Centered",
    # Head-to-head
    "A_H2H_Rate",
    # Contexte
    "Diff_Age", "Tour_Rank", "Is_K1", "Is_Male",
    "Same_Nation", "Same_Continent", "Same_Style",
    "Home_Advantage",
    # Régularité
    "Diff_Note_Std",
    # Drapeaux historiques
    "Diff_Flag_Mean",
    # Transfert
    "Transfer_Score",
    # Interactions
    "Ranking_x_IsMale", "Winrate_x_IsMale",
    "Ranking_x_Tour", "Winrate_x_Tour", "Form_x_Tour",
]

# Features pour modèles séparés (sans Is_K1 car implicite)
FEATURE_COLS_SPLIT = [c for c in FEATURE_COLS if c != "Is_K1"]


# ══════════════════════════════════════════════════════════════════════════════
# 7. PIPELINE PRINCIPALE
# ══════════════════════════════════════════════════════════════════════════════

def get_model_data(split_by_type: bool = True):
    """
    Pipeline complète.
    
    Args:
        split_by_type: si True, retourne des données séparées K1 et SA
    
    Returns:
        Si split_by_type=True:
            dict avec clés "K1" et "SA", chacune contenant (train_X, train_y, test_X, test_y)
        Si split_by_type=False:
            (train_X, train_y, test_X, test_y, full_df)
    """
    print("📦 Chargement des données...")
    raw = load_raw_data()

    print("🔗 Appairage des matchs...")
    matches = pair_matches(raw)
    print(f"   {len(matches)} matchs appairés (dont {matches['A_Drapeau'].notna().sum()} avec drapeaux)")

    print("⚙️  Feature engineering (cumulatif, anti-leakage)...")
    featured = engineer_features(matches)
    featured = add_transfer_features(featured)

    print("🎯 Construction de la target...")
    with_target = build_target(featured)
    print(f"   {len(with_target)} matchs avec drapeaux valides")

    if split_by_type:
        results = {}
        for type_name in ["K1", "SA"]:
            subset = with_target[with_target["Type_Compet"] == type_name].copy()
            if len(subset) < 10:
                print(f"   ⚠️  {type_name}: seulement {len(subset)} matchs, skip")
                continue

            train, test = split_chronological(subset)
            cols = FEATURE_COLS_SPLIT
            train_X = train[cols].fillna(0).astype(float)
            test_X = test[cols].fillna(0).astype(float)
            train_y = train["Target"].values.astype(int)
            test_y = test["Target"].values.astype(int)

            print(f"   {type_name}: Train={len(train)} | Test={len(test)}")
            results[type_name] = {
                "train_X": train_X, "train_y": train_y,
                "test_X": test_X, "test_y": test_y,
                "train_df": train, "test_df": test,
            }
        return results
    else:
        train, test = split_chronological(with_target)
        cols = FEATURE_COLS
        train_X = train[cols].fillna(0).astype(float)
        test_X = test[cols].fillna(0).astype(float)
        train_y = train["Target"].values.astype(int)
        test_y = test["Target"].values.astype(int)
        print(f"   Global: Train={len(train)} | Test={len(test)}")
        return train_X, train_y, test_X, test_y, with_target


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Test en mode séparé K1/SA
    data = get_model_data(split_by_type=True)

    print("\n" + "=" * 60)
    print("RÉSUMÉ PIPELINE V2")
    print("=" * 60)
    print(f"\nFeatures ({len(FEATURE_COLS_SPLIT)}):")
    for f in FEATURE_COLS_SPLIT:
        print(f"  • {f}")

    for type_name, d in data.items():
        print(f"\n{'─' * 40}")
        print(f"  {type_name} — Train: {d['train_X'].shape} | Test: {d['test_X'].shape}")
        print(f"  Distribution target (Train):")
        print(f"    {pd.Series(d['train_y']).value_counts().sort_index().to_dict()}")
        print(f"  Distribution target (Test):")
        print(f"    {pd.Series(d['test_y']).value_counts().sort_index().to_dict()}")
