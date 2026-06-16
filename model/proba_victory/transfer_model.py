"""
Modèle de Transfert : Notes → Score de dominance
=================================================

Objectif : entraîner un modèle sur les matchs 2024-2025 (qui ont des notes)
pour prédire un "score de dominance" (marge normalisée de victoire).

Ce modèle est ensuite utilisé comme feature (Transfer_Score) dans le modèle
de prédiction aux drapeaux 2026+.

Sera AUSSI réutilisé pour le modèle de prédiction par kata.

Architecture :
- Input : features des matchs (ranking diff, winrate, kata WR, etc.)
- Target : Note_Diff normalisée (proxy de la marge de dominance)
- Output : un score continu ∈ [-1, 1] prédisant la dominance de A sur B

Le modèle apprend sur 2024-2025 la relation features → marge,
puis on l'applique sur 2026 pour avoir un "Transfer_Score" même sans notes.
"""

import sys
from pathlib import Path
import pickle
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_data import load_raw_data, pair_matches, get_compet_chrono_rank

OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1. PRÉPARATION DONNÉES TRANSFERT (matchs avec notes = 2024-2025)
# ══════════════════════════════════════════════════════════════════════════════

# Features utilisables pour le transfert (pas de features drapeau !)
TRANSFER_FEATURES = [
    "Diff_Ranking", "Log_Ranking_Ratio", "Diff_Age",
    "Diff_Winrate", "Diff_Form", "Diff_Kata_WR",
    "A_H2H_Rate", "Exp_Diff",
    "Same_Nation", "Same_Continent", "Same_Style",
    "Tour_Rank", "Is_K1", "Is_Male",
    "Home_Advantage", "Diff_Kata_Diversity",
    # Interactions
    "Ranking_x_Tour", "Ranking_x_IsMale",
]


def build_transfer_dataset(matches: pd.DataFrame):
    """
    Construit le dataset de transfert à partir de matchs avec notes.
    
    Target : Note_Diff normalisée = (A_Note - B_Note) / std_note
    Filtrage : matchs où les deux athlètes ont une note.
    """
    # Filtrer matchs avec notes pour les deux athlètes
    df = matches[matches["A_Note"].notna() & matches["B_Note"].notna()].copy()
    
    if df.empty:
        raise ValueError("Aucun match avec notes pour les deux athlètes")
    
    # Target : note diff normalisée
    note_std = df["A_Note"].std()
    if note_std == 0:
        note_std = 1.0
    df["Target_Dominance"] = (df["A_Note"] - df["B_Note"]) / note_std
    
    # Clip pour éviter les outliers extrêmes
    df["Target_Dominance"] = df["Target_Dominance"].clip(-3, 3)
    
    print(f"  📊 Dataset transfert: {len(df)} matchs avec notes")
    print(f"     Target (dominance) — mean: {df['Target_Dominance'].mean():.3f}, "
          f"std: {df['Target_Dominance'].std():.3f}")
    
    return df, note_std


def compute_transfer_features(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les features pour le modèle de transfert.
    Même logique cumulative que prepare_data mais SANS les features drapeaux.
    """
    df = matches.copy()
    
    # ── Features statiques ──
    df["Diff_Ranking"] = df["B_Ranking"] - df["A_Ranking"]
    df["Log_Ranking_Ratio"] = np.log1p(df["B_Ranking"]) - np.log1p(df["A_Ranking"])
    df["Diff_Age"] = (df["A_Age"] - df["B_Age"]) / 10.0
    df["Same_Nation"] = (df["A_Nation"] == df["B_Nation"]).astype(int)
    df["Same_Continent"] = (df["A_Continent"] == df["B_Continent"]).astype(int)
    df["Same_Style"] = (df["A_Style"] == df["B_Style"]).astype(int)
    df["Is_K1"] = (df["Type_Compet"] == "K1").astype(int)
    df["Is_Male"] = (df["Sexe"] == "M").astype(int)
    
    tour_order = {
        "Pool_1": 1, "Pool_2": 2, "Pool_3": 3,
        "T1": 1, "T2": 2, "T3": 3,
        "PW1": 4, "PW2": 5, "PW3": 6,
        "RP1": 4, "RP2": 5, "RP3": 6, "RP4": 7,
        "R1": 7, "R2": 8, "Bronze": 9, "Final": 10,
    }
    df["Tour_Rank"] = df["N_Tour"].map(tour_order).fillna(5) / 10.0
    
    df["Is_Home_A"] = (df["A_Continent"].astype(str) == df.get("A_Region", "").astype(str)).astype(int)
    df["Is_Home_B"] = (df["B_Continent"].astype(str) == df.get("B_Region", "").astype(str)).astype(int)
    df["Home_Advantage"] = df["Is_Home_A"] - df["Is_Home_B"]
    
    # ── Features historiques cumulatives ──
    athlete_wins = {}
    athlete_recent = {}
    athlete_kata_wins = {}
    athlete_katas_used = {}
    h2h_record = {}
    
    feats = {
        "Diff_Winrate": [], "Diff_Form": [], "A_H2H_Rate": [],
        "Diff_Kata_WR": [], "Exp_Diff": [], "Diff_Kata_Diversity": [],
    }
    
    for _, row in df.iterrows():
        a, b = row["A_Nom"], row["B_Nom"]
        a_kata, b_kata = row["A_Kata"], row["B_Kata"]
        
        a_stats = athlete_wins.get(a, [0, 0])
        b_stats = athlete_wins.get(b, [0, 0])
        
        a_wr = a_stats[0] / a_stats[1] if a_stats[1] > 0 else 0.5
        b_wr = b_stats[0] / b_stats[1] if b_stats[1] > 0 else 0.5
        feats["Diff_Winrate"].append(a_wr - b_wr)
        
        a_rec = athlete_recent.get(a, [])
        b_rec = athlete_recent.get(b, [])
        a_form = np.mean(a_rec[-10:]) if a_rec else 0.5
        b_form = np.mean(b_rec[-10:]) if b_rec else 0.5
        feats["Diff_Form"].append(a_form - b_form)
        
        key = frozenset([a, b])
        if key in h2h_record:
            rec = h2h_record[key]
            total = sum(rec.values())
            feats["A_H2H_Rate"].append(rec.get(a, 0) / total if total > 0 else 0.5)
        else:
            feats["A_H2H_Rate"].append(0.5)
        
        ak = athlete_kata_wins.get((a, a_kata), [0, 0])
        bk = athlete_kata_wins.get((b, b_kata), [0, 0])
        feats["Diff_Kata_WR"].append(
            ((ak[0]+1)/(ak[1]+2)) - ((bk[0]+1)/(bk[1]+2))
        )
        
        feats["Exp_Diff"].append(np.log1p(a_stats[1]) - np.log1p(b_stats[1]))
        
        a_katas = athlete_katas_used.get(a, set())
        b_katas = athlete_katas_used.get(b, set())
        a_div = len(a_katas) / max(a_stats[1], 1)
        b_div = len(b_katas) / max(b_stats[1], 1)
        feats["Diff_Kata_Diversity"].append(a_div - b_div)
        
        # Update
        win_a = int(row["A_Win"])
        for nom, won in [(a, win_a), (b, 1 - win_a)]:
            if nom not in athlete_wins:
                athlete_wins[nom] = [0, 0]
            athlete_wins[nom][1] += 1
            athlete_wins[nom][0] += won
            if nom not in athlete_recent:
                athlete_recent[nom] = []
            athlete_recent[nom].append(won)
        
        if key not in h2h_record:
            h2h_record[key] = {}
        winner = a if win_a else b
        h2h_record[key][winner] = h2h_record[key].get(winner, 0) + 1
        
        for nom, kata, won in [(a, a_kata, win_a), (b, b_kata, 1 - win_a)]:
            if (nom, kata) not in athlete_kata_wins:
                athlete_kata_wins[(nom, kata)] = [0, 0]
            athlete_kata_wins[(nom, kata)][1] += 1
            athlete_kata_wins[(nom, kata)][0] += won
            if nom not in athlete_katas_used:
                athlete_katas_used[nom] = set()
            athlete_katas_used[nom].add(kata)
    
    for col, values in feats.items():
        df[col] = values
    
    # ── Interactions ──
    df["Ranking_x_Tour"] = df["Log_Ranking_Ratio"] * df["Tour_Rank"]
    df["Ranking_x_IsMale"] = df["Log_Ranking_Ratio"] * df["Is_Male"]
    
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. ENTRAÎNEMENT DU MODÈLE DE TRANSFERT
# ══════════════════════════════════════════════════════════════════════════════

def train_transfer_model():
    """
    Entraîne le modèle de transfert sur les matchs avec notes (2024-2025).
    Sauvegarde le modèle pour réutilisation.
    """
    print("=" * 60)
    print("MODÈLE DE TRANSFERT : Notes → Score de Dominance")
    print("=" * 60)
    
    # Charger et appairer TOUS les matchs
    print("\n📦 Chargement...")
    raw = load_raw_data()
    matches = pair_matches(raw)
    print(f"   {len(matches)} matchs totaux")
    
    # Calculer les features de transfert
    print("⚙️  Calcul features transfert...")
    featured = compute_transfer_features(matches)
    
    # Construire le dataset transfert (matchs avec notes)
    print("🎯 Construction dataset transfert...")
    transfer_df, note_std = build_transfer_dataset(featured)
    
    # Features et target
    X = transfer_df[TRANSFER_FEATURES].fillna(0).astype(float)
    y = transfer_df["Target_Dominance"].values
    
    # Standardisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # ── Cross-validation pour évaluer ──
    print("\n🔍 Cross-validation (5-fold)...")
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=15,
        subsample=0.8,
        random_state=42,
    )
    
    cv_scores_r2 = cross_val_score(model, X_scaled, y, cv=5, scoring="r2")
    cv_scores_mae = -cross_val_score(model, X_scaled, y, cv=5, scoring="neg_mean_absolute_error")
    
    print(f"   R² CV:  {cv_scores_r2.mean():.4f} ± {cv_scores_r2.std():.4f}")
    print(f"   MAE CV: {cv_scores_mae.mean():.4f} ± {cv_scores_mae.std():.4f}")
    
    # ── Entraînement final sur toutes les données notes ──
    print("\n🔧 Entraînement final sur tout le dataset notes...")
    model.fit(X_scaled, y)
    
    # Évaluation sur train (pour info)
    y_pred_train = model.predict(X_scaled)
    r2_train = r2_score(y, y_pred_train)
    mae_train = mean_absolute_error(y, y_pred_train)
    print(f"   R² train: {r2_train:.4f}")
    print(f"   MAE train: {mae_train:.4f}")
    
    # Feature importance
    feat_imp = pd.DataFrame({
        "Feature": TRANSFER_FEATURES,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)
    
    print(f"\n📋 Feature Importance (transfert) :")
    print(feat_imp.to_string(index=False))
    
    # ── Sauvegarde ──
    transfer_data = {
        "model": model,
        "scaler": scaler,
        "note_std": note_std,
        "features": TRANSFER_FEATURES,
        "cv_r2": float(cv_scores_r2.mean()),
        "cv_mae": float(cv_scores_mae.mean()),
    }
    
    with open(OUTPUT_DIR / "transfer_model.pkl", "wb") as f:
        pickle.dump(transfer_data, f)
    
    results = {
        "cv_r2_mean": float(cv_scores_r2.mean()),
        "cv_r2_std": float(cv_scores_r2.std()),
        "cv_mae_mean": float(cv_scores_mae.mean()),
        "train_r2": float(r2_train),
        "train_mae": float(mae_train),
        "n_samples": len(y),
        "note_std": float(note_std),
    }
    
    with open(OUTPUT_DIR / "transfer_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Modèle de transfert sauvegardé dans {OUTPUT_DIR}/transfer_model.pkl")
    print(f"   (à réutiliser pour le modèle kata également)")
    
    return model, scaler, note_std


def load_transfer_model():
    """Charge le modèle de transfert sauvegardé."""
    path = OUTPUT_DIR / "transfer_model.pkl"
    if not path.exists():
        print("⚠️  Modèle de transfert non trouvé, entraînement...")
        train_transfer_model()
    
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["scaler"], data["features"], data["note_std"]


def predict_dominance(matches_featured: pd.DataFrame) -> np.ndarray:
    """
    Applique le modèle de transfert pour prédire le score de dominance
    sur N'IMPORTE quel match (même sans notes).
    
    C'est la fonction clé : elle génère le Transfer_Score.
    """
    model, scaler, features, _ = load_transfer_model()
    
    X = matches_featured[features].fillna(0).astype(float)
    X_scaled = scaler.transform(X)
    
    return model.predict(X_scaled)


if __name__ == "__main__":
    train_transfer_model()
