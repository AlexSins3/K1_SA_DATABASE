"""
Préparation des données pour le modèle de victoire par kata.
=============================================================

Objectif : prédire P(A gagne | kata_A, contexte)
C'est un problème de classification binaire (win/loss) conditionné sur le kata choisi.

Pipeline : CSV → matches appairés → directed rows → features → train/test split
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "Database_K1_SA.csv"
OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

TOUR_ORDER = {
    "Pool_1": 1, "Pool_2": 2, "Pool_3": 3, "Pool_4": 4,
    "T1": 1, "T2": 2, "T3": 3,
    "PW1": 4, "PW2": 5, "PW3": 6,
    "RP1": 4, "RP2": 5, "RP3": 6, "RP4": 7,
    "R1": 7, "R2": 8,
    "Bronze": 9, "Final": 10,
}

# Compétitions en ordre chronologique (2024-2026)
COMPET_CHRONO = {
    "K1 Istanbul": 1, "SA Pamplona": 2, "K1 Cairo": 3, "SA Porec": 4,
    "K1 Shanghai": 5, "SA Kocaeli": 6, "K1 Rabat": 7, "SA Rabat": 8,
    "K1 Budapest": 9, "SA Budapest": 10, "K1 Konya": 11,
    "K1_Istanbul": 1, "SA_Pamplona": 2, "K1_Cairo": 3, "SA_Porec": 4,
    "K1_Shanghai": 5, "SA_Kocaeli": 6, "K1_Rabat": 7, "SA_Rabat": 8,
    "K1_Budapest": 9, "SA_Budapest": 10, "K1_Konya": 11,
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD & PAIR MATCHES
# ══════════════════════════════════════════════════════════════════════════════

def load_raw_data() -> pd.DataFrame:
    """Charge le CSV brut."""
    df = pd.read_csv(DATA_PATH, sep=";")
    for col in ["Age", "Ranking", "Note", "Year", "Drapeau"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Victoire → bool
    df["Victoire_bool"] = df["Victoire"].astype(str).str.lower().isin(["true", "1", "vrai", "yes"])
    return df


def pair_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Appaire les matchs (stride-2 + validation victoire)."""
    d = df.sort_values(
        ["Year", "Competition", "Type_Compet", "N_Tour"], kind="mergesort"
    ).reset_index(drop=True)

    pairs = []
    i = 0
    while i < len(d) - 1:
        row1 = d.iloc[i]
        row2 = d.iloc[i + 1]
        is_pair = (
            row1["Competition"] == row2["Competition"]
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
        # Red = A
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
        # Blue = B
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
    matches["chrono_rank"] = matches["Competition"].astype(str).map(COMPET_CHRONO).fillna(5)
    matches = matches.sort_values("chrono_rank").reset_index(drop=True)
    return matches


# ══════════════════════════════════════════════════════════════════════════════
# 2. DIRECTED ROWS (chaque match donne 2 lignes: A→B et B→A)
# ══════════════════════════════════════════════════════════════════════════════

def to_directed(matches: pd.DataFrame) -> pd.DataFrame:
    """Déplie chaque match en 2 lignes (perspective de chaque athlète)."""
    if matches.empty:
        return pd.DataFrame()

    rows_a = pd.DataFrame({
        "Competition": matches["Competition"],
        "Year": matches["Year"],
        "Type_Compet": matches["Type_Compet"],
        "N_Tour": matches["N_Tour"],
        "Sexe": matches["Sexe"],
        "chrono_rank": matches["chrono_rank"],
        "Nom": matches["A_Nom"],
        "Kata": matches["A_Kata"],
        "Style": matches["A_Style"],
        "Nation": matches["A_Nation"],
        "Continent": matches["A_Continent"],
        "Region": matches["A_Region"],
        "Ranking": matches["A_Ranking"],
        "Age": matches["A_Age"],
        "Note": matches["A_Note"],
        "Drapeau": matches["A_Drapeau"],
        "Opponent": matches["B_Nom"],
        "Opp_Kata": matches["B_Kata"],
        "Opp_Style": matches["B_Style"],
        "Opp_Nation": matches["B_Nation"],
        "Opp_Ranking": matches["B_Ranking"],
        "Opp_Age": matches["B_Age"],
        "Opp_Note": matches["B_Note"],
        "Opp_Continent": matches["B_Continent"],
        "Win": matches["A_Win"].astype(int),
        "Is_Red": 1,
    })

    rows_b = pd.DataFrame({
        "Competition": matches["Competition"],
        "Year": matches["Year"],
        "Type_Compet": matches["Type_Compet"],
        "N_Tour": matches["N_Tour"],
        "Sexe": matches["Sexe"],
        "chrono_rank": matches["chrono_rank"],
        "Nom": matches["B_Nom"],
        "Kata": matches["B_Kata"],
        "Style": matches["B_Style"],
        "Nation": matches["B_Nation"],
        "Continent": matches["B_Continent"],
        "Region": matches["B_Region"],
        "Ranking": matches["B_Ranking"],
        "Age": matches["B_Age"],
        "Note": matches["B_Note"],
        "Drapeau": matches["B_Drapeau"],
        "Opponent": matches["A_Nom"],
        "Opp_Kata": matches["A_Kata"],
        "Opp_Style": matches["A_Style"],
        "Opp_Nation": matches["A_Nation"],
        "Opp_Ranking": matches["A_Ranking"],
        "Opp_Age": matches["A_Age"],
        "Opp_Note": matches["A_Note"],
        "Opp_Continent": matches["A_Continent"],
        "Win": (1 - matches["A_Win"]).astype(int),
        "Is_Red": 0,
    })

    directed = pd.concat([rows_a, rows_b], ignore_index=True)
    directed = directed.sort_values("chrono_rank").reset_index(drop=True)
    return directed


# ══════════════════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING (cumulatif, anti-leakage)
# ══════════════════════════════════════════════════════════════════════════════

def engineer_features(directed: pd.DataFrame) -> pd.DataFrame:
    """
    Calcul cumulatif des features AVANT chaque match (anti-leakage).
    Chaque feature est calculée à partir de l'historique AVANT ce match.
    """
    df = directed.copy()
    n = len(df)

    # Structures cumulatives
    athlete_wins = {}       # nom → [wins, total]
    athlete_notes = {}      # nom → [notes]
    athlete_recent = {}     # nom → [last 15 results]
    athlete_recent_5 = {}   # nom → [last 5 results]
    athlete_kata_wins = {}  # (nom, kata) → [wins, total]
    athlete_kata_notes = {} # (nom, kata) → [notes]
    h2h_record = {}         # frozenset → {nom: wins}
    kata_wins = {}          # kata → [wins, total]
    kata_tour_wins = {}     # (kata, tour) → [wins, total]
    athlete_katas_used = {} # nom → set
    athlete_flags = {}      # nom → [flags obtained]
    athlete_streak = {}     # nom → current_streak (+ = win, - = loss)
    opp_kata_losses = {}    # (nom, kata_faced) → [losses, total]

    # Feature arrays
    feats = {
        "WinRate_Adv": np.zeros(n),
        "A_Kata_WR": np.zeros(n),
        "B_Kata_WR": np.zeros(n),
        "A_Kata_Note_C": np.zeros(n),
        "Note_Diff": np.zeros(n),
        "Ranking_Adv": np.zeros(n),
        "H2H_Adv": np.zeros(n),
        "B_Weakness_C": np.zeros(n),
        "Kata_Tour_C": np.zeros(n),
        "Kata_Effect": np.zeros(n),
        "Log_A_Kata_N": np.zeros(n),
        "Note_Trend_Diff": np.zeros(n),
        "Momentum_Diff": np.zeros(n),
        "Recent_5_Diff": np.zeros(n),
        "Age_Diff": np.zeros(n),
        "Is_Fav_Kata_A": np.zeros(n),
        "Note_Std_Diff": np.zeros(n),
        "Exp_Diff": np.zeros(n),
        "Kata_Diversity_Diff": np.zeros(n),
        "B_Seen_Kata": np.zeros(n),
        "Flag_WR_Diff": np.zeros(n),
        "Win_Streak_Diff": np.zeros(n),
        # New features
        "Transfer_Score": np.zeros(n),
        "A_Kata_Consistency": np.zeros(n),
        "Opp_Style_WR": np.zeros(n),
    }

    # Load transfer model if available
    transfer_model_path = Path(__file__).parent.parent / "proba_victory" / "results" / "transfer_model.pkl"
    transfer_model = None
    if transfer_model_path.exists():
        import pickle
        with open(transfer_model_path, "rb") as f:
            transfer_data = pickle.load(f)
        transfer_model = transfer_data["model"]
        transfer_features = transfer_data["features"]

    # Global note mean (computed first)
    all_notes = df["Note"].dropna()
    global_note_mean = float(all_notes.mean()) if len(all_notes) > 0 else 24.0

    # Style win rates (pre-computed on full data for simplicity — minor leakage but stable)
    style_wr = {}
    for style in df["Style"].dropna().unique():
        s_data = df[df["Style"] == style]
        if len(s_data) >= 5:
            style_wr[str(style)] = float(s_data["Win"].mean())
        else:
            style_wr[str(style)] = 0.5

    for idx in range(n):
        row = df.iloc[idx]
        nom = str(row["Nom"])
        opp = str(row["Opponent"])
        kata = str(row["Kata"])
        opp_kata = str(row["Opp_Kata"])
        tour = str(row["N_Tour"])
        win = int(row["Win"])

        # ── FEATURES (computed from state BEFORE this match) ──

        # WinRate
        a_wr = (athlete_wins.get(nom, [0, 0])[0] + 2) / (athlete_wins.get(nom, [0, 0])[1] + 4)
        b_wr = (athlete_wins.get(opp, [0, 0])[0] + 2) / (athlete_wins.get(opp, [0, 0])[1] + 4)
        feats["WinRate_Adv"][idx] = a_wr - b_wr

        # Kata win rate
        a_kw = athlete_kata_wins.get((nom, kata), [0, 0])
        feats["A_Kata_WR"][idx] = (a_kw[0] + 1) / (a_kw[1] + 2) - 0.5
        feats["Log_A_Kata_N"][idx] = np.log1p(a_kw[1])

        b_kw = athlete_kata_wins.get((opp, opp_kata), [0, 0])
        feats["B_Kata_WR"][idx] = (b_kw[0] + 1) / (b_kw[1] + 2) - 0.5

        # A kata note centered
        a_kata_notes_list = athlete_kata_notes.get((nom, kata), [])
        a_kata_note_mean = np.mean(a_kata_notes_list) if a_kata_notes_list else global_note_mean
        feats["A_Kata_Note_C"][idx] = a_kata_note_mean - global_note_mean

        # Note diff
        b_notes_list = athlete_notes.get(opp, [])
        b_note_mean = np.mean(b_notes_list) if b_notes_list else global_note_mean
        feats["Note_Diff"][idx] = a_kata_note_mean - b_note_mean

        # Ranking
        a_rank = float(row["Ranking"]) if pd.notna(row["Ranking"]) else 50.0
        b_rank = float(row["Opp_Ranking"]) if pd.notna(row["Opp_Ranking"]) else 50.0
        feats["Ranking_Adv"][idx] = (b_rank - a_rank) / 100.0

        # H2H
        key = frozenset([nom, opp])
        h2h_data = h2h_record.get(key, {})
        h2h_total = sum(h2h_data.values())
        h2h_wr = (h2h_data.get(nom, 0) + 1) / (h2h_total + 2)
        feats["H2H_Adv"][idx] = (h2h_wr - 0.5) * np.log1p(h2h_total)

        # B weakness vs A's kata
        opp_kata_data = opp_kata_losses.get((opp, kata), [0, 0])
        b_loss_rate = (opp_kata_data[0] + 1) / (opp_kata_data[1] + 2)
        feats["B_Weakness_C"][idx] = b_loss_rate - 0.5

        # Kata × tour WR
        kt_data = kata_tour_wins.get((kata, tour), [0, 0])
        feats["Kata_Tour_C"][idx] = (kt_data[0] + 1) / (kt_data[1] + 2) - 0.5

        # Kata global effect
        k_data = kata_wins.get(kata, [0, 0])
        feats["Kata_Effect"][idx] = ((k_data[0] + 1) / (k_data[1] + 2) - 0.5) if k_data[1] > 0 else 0.0

        # Note trend
        a_notes_list = athlete_notes.get(nom, [])
        b_notes_list_full = athlete_notes.get(opp, [])
        a_trend = 0.0
        if len(a_notes_list) >= 4:
            third = len(a_notes_list) // 3
            a_trend = np.mean(a_notes_list[-third:]) - np.mean(a_notes_list[:third])
        b_trend = 0.0
        if len(b_notes_list_full) >= 4:
            third = len(b_notes_list_full) // 3
            b_trend = np.mean(b_notes_list_full[-third:]) - np.mean(b_notes_list_full[:third])
        feats["Note_Trend_Diff"][idx] = a_trend - b_trend

        # Momentum (recent 15)
        a_rec = athlete_recent.get(nom, [])
        b_rec = athlete_recent.get(opp, [])
        a_momentum = np.mean(a_rec[-15:]) if len(a_rec) >= 3 else 0.5
        b_momentum = np.mean(b_rec[-15:]) if len(b_rec) >= 3 else 0.5
        feats["Momentum_Diff"][idx] = a_momentum - b_momentum

        # Recent 5
        a_r5 = athlete_recent_5.get(nom, [])
        b_r5 = athlete_recent_5.get(opp, [])
        a_r5_wr = np.mean(a_r5[-5:]) if len(a_r5) >= 2 else 0.5
        b_r5_wr = np.mean(b_r5[-5:]) if len(b_r5) >= 2 else 0.5
        feats["Recent_5_Diff"][idx] = a_r5_wr - b_r5_wr

        # Age
        a_age = float(row["Age"]) if pd.notna(row["Age"]) else 25.0
        b_age = float(row["Opp_Age"]) if pd.notna(row["Opp_Age"]) else 25.0
        feats["Age_Diff"][idx] = (a_age - b_age) / 10.0

        # Favourite kata
        a_katas = athlete_katas_used.get(nom, set())
        if a_katas:
            kata_counts = {k: athlete_kata_wins.get((nom, k), [0, 0])[1] for k in a_katas}
            fav = max(kata_counts, key=kata_counts.get) if kata_counts else ""
            feats["Is_Fav_Kata_A"][idx] = 1.0 if kata == fav else 0.0

        # Note std diff
        a_std = np.std(a_notes_list) if len(a_notes_list) > 2 else 0.0
        b_std = np.std(b_notes_list_full) if len(b_notes_list_full) > 2 else 0.0
        feats["Note_Std_Diff"][idx] = a_std - b_std

        # Experience diff
        a_total = athlete_wins.get(nom, [0, 0])[1]
        b_total = athlete_wins.get(opp, [0, 0])[1]
        feats["Exp_Diff"][idx] = np.log1p(a_total) - np.log1p(b_total)

        # Kata diversity
        a_div = len(athlete_katas_used.get(nom, set()))
        b_div = len(athlete_katas_used.get(opp, set()))
        feats["Kata_Diversity_Diff"][idx] = (a_div - b_div) / 5.0

        # B seen kata
        feats["B_Seen_Kata"][idx] = 1.0 if opp_kata_losses.get((opp, kata), [0, 0])[1] >= 2 else 0.0

        # Flag WR diff
        a_flags_list = athlete_flags.get(nom, [])
        b_flags_list = athlete_flags.get(opp, [])
        a_flag_wr = np.mean(a_flags_list) / 7.0 if a_flags_list else 0.5
        b_flag_wr = np.mean(b_flags_list) / 7.0 if b_flags_list else 0.5
        feats["Flag_WR_Diff"][idx] = a_flag_wr - b_flag_wr

        # Win streak
        a_streak_val = athlete_streak.get(nom, 0)
        b_streak_val = athlete_streak.get(opp, 0)
        feats["Win_Streak_Diff"][idx] = (a_streak_val - b_streak_val) / 10.0

        # Transfer Score (from proba_victory transfer model)
        if transfer_model is not None:
            log_rank_ratio = np.log1p(b_rank) - np.log1p(a_rank)
            tour_rank = TOUR_ORDER.get(tour, 5) / 10.0
            is_male = 1 if str(row.get("Sexe", "")) == "M" else 0
            is_k1 = 1 if str(row["Type_Compet"]) == "K1" else 0
            same_nation = int(str(row["Nation"]) == str(row["Opp_Nation"]))
            same_continent = int(str(row["Continent"]) == str(row.get("Opp_Continent", "")))
            same_style = int(str(row["Style"]) == str(row["Opp_Style"]))
            home_adv = int(str(row["Continent"]) == str(row["Region"]))
            diff_kata_div = (a_div - b_div) / 5.0

            tf_dict = {
                "Diff_Ranking": b_rank - a_rank,
                "Log_Ranking_Ratio": log_rank_ratio,
                "Diff_Age": (a_age - b_age) / 10.0,
                "Diff_Winrate": a_wr - b_wr,
                "Diff_Form": a_momentum - b_momentum,
                "Diff_Kata_WR": feats["A_Kata_WR"][idx],
                "A_H2H_Rate": h2h_wr,
                "Exp_Diff": feats["Exp_Diff"][idx],
                "Same_Nation": same_nation,
                "Same_Continent": same_continent,
                "Same_Style": same_style,
                "Tour_Rank": tour_rank,
                "Is_K1": is_k1,
                "Is_Male": is_male,
                "Home_Advantage": home_adv,
                "Diff_Kata_Diversity": diff_kata_div,
                "Ranking_x_Tour": log_rank_ratio * tour_rank,
                "Ranking_x_IsMale": log_rank_ratio * is_male,
            }
            tf_X = pd.DataFrame([tf_dict])[transfer_features]
            feats["Transfer_Score"][idx] = float(transfer_model.predict(tf_X)[0])

        # A Kata consistency (std of notes with this kata)
        if len(a_kata_notes_list) > 2:
            feats["A_Kata_Consistency"][idx] = -np.std(a_kata_notes_list)  # Neg = more consistent = better
        else:
            feats["A_Kata_Consistency"][idx] = 0.0

        # Opp style WR (A's win rate against this style)
        opp_style = str(row["Opp_Style"])
        feats["Opp_Style_WR"][idx] = style_wr.get(opp_style, 0.5) - 0.5

        # ── UPDATE STATE (after computing features) ──
        # Wins
        if nom not in athlete_wins:
            athlete_wins[nom] = [0, 0]
        athlete_wins[nom][1] += 1
        athlete_wins[nom][0] += win

        # Notes
        note = row["Note"]
        if pd.notna(note):
            if nom not in athlete_notes:
                athlete_notes[nom] = []
            athlete_notes[nom].append(float(note))
            if (nom, kata) not in athlete_kata_notes:
                athlete_kata_notes[(nom, kata)] = []
            athlete_kata_notes[(nom, kata)].append(float(note))

        # Kata wins
        if (nom, kata) not in athlete_kata_wins:
            athlete_kata_wins[(nom, kata)] = [0, 0]
        athlete_kata_wins[(nom, kata)][1] += 1
        athlete_kata_wins[(nom, kata)][0] += win

        # Recent
        if nom not in athlete_recent:
            athlete_recent[nom] = []
        athlete_recent[nom].append(win)
        if nom not in athlete_recent_5:
            athlete_recent_5[nom] = []
        athlete_recent_5[nom].append(win)

        # H2H
        if key not in h2h_record:
            h2h_record[key] = {}
        if win:
            h2h_record[key][nom] = h2h_record[key].get(nom, 0) + 1

        # Kata diversity
        if nom not in athlete_katas_used:
            athlete_katas_used[nom] = set()
        athlete_katas_used[nom].add(kata)

        # Kata global
        if kata not in kata_wins:
            kata_wins[kata] = [0, 0]
        kata_wins[kata][1] += 1
        kata_wins[kata][0] += win

        # Kata × tour
        if (kata, tour) not in kata_tour_wins:
            kata_tour_wins[(kata, tour)] = [0, 0]
        kata_tour_wins[(kata, tour)][1] += 1
        kata_tour_wins[(kata, tour)][0] += win

        # Flags
        drapeau = row["Drapeau"]
        if pd.notna(drapeau):
            if nom not in athlete_flags:
                athlete_flags[nom] = []
            athlete_flags[nom].append(float(drapeau))

        # Win streak
        if nom not in athlete_streak:
            athlete_streak[nom] = 0
        if win:
            athlete_streak[nom] = max(0, athlete_streak[nom]) + 1
        else:
            athlete_streak[nom] = min(0, athlete_streak[nom]) - 1

        # Opponent seen kata (B faces kata from A)
        if (opp, kata) not in opp_kata_losses:
            opp_kata_losses[(opp, kata)] = [0, 0]
        opp_kata_losses[(opp, kata)][1] += 1
        if win:  # A won → B lost against this kata
            opp_kata_losses[(opp, kata)][0] += 1

    # Assign features to dataframe
    for col, values in feats.items():
        df[col] = values

    # Static features
    df["Same_Nation"] = (df["Nation"].astype(str) == df["Opp_Nation"].astype(str)).astype(int)
    df["Same_Style"] = (df["Style"].astype(str) == df["Opp_Style"].astype(str)).astype(int)
    df["Is_K1"] = (df["Type_Compet"].astype(str) == "K1").astype(int)
    df["Is_Male"] = (df["Sexe"].astype(str) == "M").astype(int)
    df["Tour_Rank"] = df["N_Tour"].astype(str).map(lambda t: TOUR_ORDER.get(t, 5) / 10.0)
    df["Is_Home"] = (df["Continent"].astype(str) == df["Region"].astype(str)).astype(int)

    # Interactions
    df["Ranking_x_Tour"] = feats["Ranking_Adv"] * df["Tour_Rank"].values
    df["WinRate_x_Tour"] = feats["WinRate_Adv"] * df["Tour_Rank"].values
    df["KataWR_x_Tour"] = feats["A_Kata_WR"] * df["Tour_Rank"].values

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 4. FEATURE DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

# Full feature set for experimentation
ALL_FEATURES = [
    # Athlete-level differentials
    "WinRate_Adv", "Ranking_Adv", "Exp_Diff", "Momentum_Diff", "Recent_5_Diff",
    "Note_Trend_Diff", "Note_Std_Diff", "Age_Diff", "Flag_WR_Diff", "Win_Streak_Diff",
    # Kata-specific (KEY features for this model)
    "A_Kata_WR", "Log_A_Kata_N", "A_Kata_Note_C", "Note_Diff",
    "A_Kata_Consistency", "Is_Fav_Kata_A",
    # Opponent-interaction
    "H2H_Adv", "B_Weakness_C", "B_Seen_Kata", "Opp_Style_WR",
    # Kata context
    "Kata_Tour_C", "Kata_Effect",
    # Match context
    "Same_Nation", "Same_Style", "Is_K1", "Is_Male", "Is_Red",
    "Is_Home", "Tour_Rank",
    # Transfer
    "Transfer_Score",
    # Interactions
    "Ranking_x_Tour", "WinRate_x_Tour", "KataWR_x_Tour",
    "Kata_Diversity_Diff",
]

# Subset without transfer (fallback)
FEATURES_NO_TRANSFER = [f for f in ALL_FEATURES if f != "Transfer_Score"]


# ══════════════════════════════════════════════════════════════════════════════
# 5. SPLIT
# ══════════════════════════════════════════════════════════════════════════════

def split_chronological(df: pd.DataFrame, test_ratio: float = 0.2):
    """Split chronologique (les matchs les plus récents en test)."""
    df = df.sort_values("chrono_rank").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_ratio))
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    return train, test


# ══════════════════════════════════════════════════════════════════════════════
# 6. PIPELINE PRINCIPALE
# ══════════════════════════════════════════════════════════════════════════════

def get_model_data(features: list = None):
    """
    Pipeline complète. Retourne (train_X, train_y, test_X, test_y, full_df).
    """
    if features is None:
        features = ALL_FEATURES

    print("📦 Chargement...")
    raw = load_raw_data()

    print("🔗 Appairage...")
    matches = pair_matches(raw)
    print(f"   {len(matches)} matchs appairés")

    print("↔️  Directed rows...")
    directed = to_directed(matches)
    print(f"   {len(directed)} lignes dirigées")

    print("⚙️  Feature engineering (cumulatif)...")
    featured = engineer_features(directed)

    # Target
    featured["y"] = featured["Win"].astype(int)

    print(f"   Features: {len(features)}")
    print(f"   Target balance: {featured['y'].mean():.3f} (should be ~0.5)")

    train, test = split_chronological(featured)
    print(f"   Train: {len(train)} | Test: {len(test)}")

    train_X = train[features].fillna(0).astype(float)
    test_X = test[features].fillna(0).astype(float)
    train_y = train["y"].values
    test_y = test["y"].values

    return train_X, train_y, test_X, test_y, featured


if __name__ == "__main__":
    train_X, train_y, test_X, test_y, _ = get_model_data()
    print(f"\n✅ Pipeline OK")
    print(f"   Train: {train_X.shape}, Test: {test_X.shape}")
    print(f"   Train balance: {train_y.mean():.3f}")
    print(f"   Test balance: {test_y.mean():.3f}")
