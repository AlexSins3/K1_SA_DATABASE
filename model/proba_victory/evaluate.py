"""
Évaluation comparative V2 — Modèles séparés K1 / SA.
=====================================================
Compare Ordinal vs LightGBM pour chaque format de compétition.
"""

import sys
from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, mean_absolute_error, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_data import get_model_data, FEATURE_COLS_SPLIT

OUTPUT_DIR = Path(__file__).parent / "results"
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_models():
    """Charge les modèles sauvegardés."""
    models = {}
    ordinal_path = OUTPUT_DIR / "ordinal_models.pkl"
    lgbm_path = OUTPUT_DIR / "lgbm_models.pkl"

    if ordinal_path.exists():
        with open(ordinal_path, "rb") as f:
            models["ordinal"] = pickle.load(f)
    if lgbm_path.exists():
        with open(lgbm_path, "rb") as f:
            models["lgbm"] = pickle.load(f)
    return models


def load_results():
    """Charge les métriques JSON."""
    results = {}
    for name in ["ordinal", "lgbm"]:
        path = OUTPUT_DIR / f"{name}_results.json"
        if path.exists():
            with open(path) as f:
                results[name] = json.load(f)
    return results


def compare_models():
    """Comparaison complète des modèles."""

    print("=" * 70)
    print("COMPARAISON V2 — MODÈLES SÉPARÉS K1 / SA")
    print("=" * 70)

    results = load_results()
    if not results:
        print("❌ Aucun résultat. Lancez d'abord train_ordinal.py et train_lgbm.py")
        return

    for type_name in ["K1", "SA"]:
        max_f = 7 if type_name == "K1" else 5
        print(f"\n{'━' * 60}")
        print(f"  {type_name} (drapeaux: 0..{max_f})")
        print(f"{'━' * 60}")

        print(f"\n  {'Métrique':<20} {'Ordinal':>12} {'LightGBM':>12} {'Meilleur':>10}")
        print(f"  {'─' * 54}")

        metrics = ["accuracy", "mae", "accuracy_pm1", "log_loss"]
        labels = ["Accuracy exacte", "MAE (flags)", "Accuracy ±1", "Log-loss"]
        higher_better = [True, False, True, False]

        for metric, label, hb in zip(metrics, labels, higher_better):
            ord_val = results.get("ordinal", {}).get(type_name, {}).get(metric)
            lgb_val = results.get("lgbm", {}).get(type_name, {}).get(metric)

            ord_str = f"{ord_val:.4f}" if isinstance(ord_val, (int, float)) and ord_val is not None else "N/A"
            lgb_str = f"{lgb_val:.4f}" if isinstance(lgb_val, (int, float)) and lgb_val is not None else "N/A"

            winner = ""
            if isinstance(ord_val, (int, float)) and isinstance(lgb_val, (int, float)):
                if ord_val is not None and lgb_val is not None:
                    if hb:
                        winner = "← Ord" if ord_val > lgb_val else "→ LGB"
                    else:
                        winner = "← Ord" if ord_val < lgb_val else "→ LGB"

            print(f"  {label:<20} {ord_str:>12} {lgb_str:>12} {winner:>10}")

    # Visualisation
    generate_plots()


def generate_plots():
    """Génère les visualisations comparatives par format."""

    models = load_models()
    if not models:
        return

    data = get_model_data(split_by_type=True)

    for type_name, d in data.items():
        test_X, test_y = d["test_X"], d["test_y"]
        max_f = 7 if type_name == "K1" else 5

        predictions = {}

        # Ordinal
        if "ordinal" in models and type_name in models["ordinal"]:
            scaler = models["ordinal"][type_name]["scaler"]
            model = models["ordinal"][type_name]["model"]
            sel_feats = models["ordinal"][type_name].get("selected_features", list(test_X.columns))
            predictions["Ordinal"] = model.predict(scaler.transform(test_X[sel_feats]))

        # LightGBM
        if "lgbm" in models and type_name in models["lgbm"]:
            model = models["lgbm"][type_name]["model"]
            predictions["LightGBM"] = model.predict(test_X)

        if not predictions:
            continue

        n_models = len(predictions)
        fig, axes = plt.subplots(2, n_models + 1, figsize=(5 * (n_models + 1), 8))
        fig.suptitle(f"{type_name} — Comparaison Modèles (drapeaux 0..{max_f})",
                     fontsize=14, fontweight="bold")

        # Distribution réelle
        ax = axes[0, 0]
        pd.Series(test_y).value_counts().reindex(range(max_f + 1), fill_value=0).plot(
            kind="bar", ax=ax, color="gray", alpha=0.7)
        ax.set_title("Réel")
        ax.set_xlabel("Drapeaux A")

        for idx, (name, pred) in enumerate(predictions.items()):
            # Distribution prédite
            ax = axes[0, idx + 1]
            pd.Series(pred).value_counts().reindex(range(max_f + 1), fill_value=0).plot(
                kind="bar", ax=ax, alpha=0.7)
            ax.set_title(f"Prédit — {name}")
            ax.set_xlabel("Drapeaux A")

            # Matrice de confusion
            ax = axes[1, idx]
            cm = confusion_matrix(test_y, pred, labels=range(max_f + 1))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=range(max_f + 1), yticklabels=range(max_f + 1))
            ax.set_title(f"Confusion — {name}")
            ax.set_xlabel("Prédit")
            ax.set_ylabel("Réel")

        # Erreurs
        ax = axes[1, n_models]
        for name, pred in predictions.items():
            errors = pred.astype(int) - test_y
            ax.hist(errors, bins=range(-max_f - 1, max_f + 2), alpha=0.5,
                    label=name, edgecolor="black")
        ax.axvline(0, color="red", linestyle="--", alpha=0.7)
        ax.set_title("Distribution erreurs")
        ax.set_xlabel("Erreur (Prédit - Réel)")
        ax.legend()

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"comparison_{type_name}.png", dpi=150, bbox_inches="tight")
        plt.close()

    print(f"\n  📊 Figures sauvegardées dans {FIGURES_DIR}/")


def print_examples():
    """Affiche des exemples de prédictions."""

    models = load_models()
    if not models:
        return

    data = get_model_data(split_by_type=True)

    print("\n" + "=" * 70)
    print("EXEMPLES DE PRÉDICTIONS")
    print("=" * 70)

    for type_name, d in data.items():
        test_X, test_y = d["test_X"], d["test_y"]
        test_df = d["test_df"]
        max_f = 7 if type_name == "K1" else 5

        print(f"\n{'─' * 50}")
        print(f"  {type_name} (max {max_f} drapeaux)")
        print(f"{'─' * 50}")

        np.random.seed(42)
        n_examples = min(8, len(test_y))
        sample_idx = np.random.choice(len(test_y), n_examples, replace=False)

        for i in sample_idx:
            row = test_df.iloc[i]
            a_nom = row["A_Nom"]
            b_nom = row["B_Nom"]
            real = test_y[i]
            real_score = f"{real}-{max_f - real}"

            preds = {}
            if "ordinal" in models and type_name in models["ordinal"]:
                scaler = models["ordinal"][type_name]["scaler"]
                model = models["ordinal"][type_name]["model"]
                sel_feats = models["ordinal"][type_name].get("selected_features", list(test_X.columns))
                p = model.predict(scaler.transform(test_X[sel_feats].iloc[[i]]))[0]
                preds["Ordinal"] = int(p)
            if "lgbm" in models and type_name in models["lgbm"]:
                model = models["lgbm"][type_name]["model"]
                p = model.predict(test_X.iloc[[i]])[0]
                preds["LightGBM"] = int(p)

            print(f"\n  {a_nom} vs {b_nom}")
            print(f"    Réel: {real_score}")
            for name, p in preds.items():
                pred_score = f"{p}-{max_f - p}"
                ok = "✓" if p == real else ("~" if abs(p - real) == 1 else "✗")
                print(f"    {name:>10}: {pred_score} {ok}")


if __name__ == "__main__":
    compare_models()
    print_examples()
