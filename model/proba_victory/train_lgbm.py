"""
Modèle 2 : LightGBM Multiclass + Optuna — V2 (séparé K1/SA)
=============================================================
Prédit le nombre de drapeaux de l'athlète A (classification multiclass).

Avantages :
- Capture les interactions non-linéaires
- Gère nativement les valeurs manquantes
- Modèles séparés K1 (0-7) et SA (0-5) optimisés indépendamment
"""

import sys
from pathlib import Path
import pickle
import json
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.metrics import (
    accuracy_score, log_loss, classification_report,
    confusion_matrix, mean_absolute_error
)
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_data import get_model_data, FEATURE_COLS_SPLIT


OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)


def objective(trial, train_X, train_y, num_classes):
    """Optuna objective : minimise log-loss en CV stratifiée."""

    params = {
        "objective": "multiclass",
        "num_class": num_classes,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "random_state": 42,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 8, 64),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 5.0, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 400),
    }

    n_splits = min(5, max(3, len(train_X) // 50))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in skf.split(train_X, train_y):
        X_tr = train_X.iloc[train_idx]
        X_val = train_X.iloc[val_idx]
        y_tr, y_val = train_y[train_idx], train_y[val_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
        )

        val_proba = model.predict_proba(X_val)
        score = log_loss(y_val, val_proba, labels=list(range(num_classes)))
        scores.append(score)

    return np.mean(scores)


def train_single_lgbm(train_X, train_y, test_X, test_y, type_name: str, n_trials: int = 80):
    """Entraîne + optimise un LightGBM pour un type de compétition."""

    max_flags = 7 if type_name == "K1" else 5
    num_classes = max_flags + 1

    print(f"\n{'═' * 50}")
    print(f"  {type_name} — LightGBM + Optuna ({n_trials} trials)")
    print(f"  Classes: 0..{max_flags} ({num_classes} classes)")
    print(f"{'═' * 50}")

    # Optimisation
    study = optuna.create_study(direction="minimize", study_name=f"lgbm_{type_name}")
    study.optimize(
        lambda trial: objective(trial, train_X, train_y, num_classes),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    print(f"\n  ✅ Best CV log-loss: {study.best_value:.4f}")
    best_params = study.best_params

    # Entraînement final
    final_params = {
        "objective": "multiclass",
        "num_class": num_classes,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "random_state": 42,
        **best_params,
    }

    model = lgb.LGBMClassifier(**final_params)
    model.fit(train_X, train_y)

    # Prédictions
    test_pred = model.predict(test_X)
    test_proba = model.predict_proba(test_X)
    train_pred = model.predict(train_X)

    # Métriques
    acc = accuracy_score(test_y, test_pred)
    mae = mean_absolute_error(test_y, test_pred)
    ll = log_loss(test_y, test_proba, labels=list(range(num_classes)))
    acc_1 = np.mean(np.abs(test_y - test_pred) <= 1)
    train_acc = accuracy_score(train_y, train_pred)

    print(f"\n  📊 Accuracy exacte:  {acc:.4f}")
    print(f"  📊 MAE (drapeaux):   {mae:.4f}")
    print(f"  📊 Accuracy ±1:      {acc_1:.4f}")
    print(f"  📊 Log-loss:         {ll:.4f}")
    print(f"  📊 Train accuracy:   {train_acc:.4f} (overfit check)")

    print(f"\n  📋 Classification Report:")
    print(classification_report(test_y, test_pred, zero_division=0))

    # Feature importance
    feat_imp = pd.DataFrame({
        "Feature": FEATURE_COLS_SPLIT,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)
    print(f"\n  📋 Top features :")
    print(feat_imp.head(10).to_string(index=False))

    return {
        "model": model, "study": study, "best_params": best_params,
        "metrics": {
            "accuracy": float(acc), "mae": float(mae),
            "accuracy_pm1": float(acc_1), "log_loss": float(ll),
            "train_accuracy": float(train_acc),
            "best_cv_logloss": float(study.best_value),
        },
        "feature_importance": feat_imp,
    }


def train_lgbm_model(n_trials: int = 80):
    """Pipeline complète : optimise + entraîne LightGBM séparé K1/SA."""

    print("=" * 60)
    print("MODÈLE 2 : LightGBM MULTICLASS + OPTUNA — V2")
    print("  (modèles séparés K1 / SA)")
    print("=" * 60)

    data = get_model_data(split_by_type=True)

    all_results = {}
    for type_name, d in data.items():
        result = train_single_lgbm(
            d["train_X"], d["train_y"], d["test_X"], d["test_y"],
            type_name, n_trials=n_trials
        )
        all_results[type_name] = result

    # Sauvegarde
    save_data = {}
    save_metrics = {}
    for type_name, r in all_results.items():
        save_data[type_name] = {"model": r["model"], "best_params": r["best_params"]}
        save_metrics[type_name] = r["metrics"]
        r["feature_importance"].to_csv(
            OUTPUT_DIR / f"lgbm_{type_name}_feature_importance.csv", index=False
        )

    with open(OUTPUT_DIR / "lgbm_models.pkl", "wb") as f:
        pickle.dump(save_data, f)

    with open(OUTPUT_DIR / "lgbm_results.json", "w") as f:
        json.dump(save_metrics, f, indent=2)

    print(f"\n✅ Modèles sauvegardés dans {OUTPUT_DIR}/")
    return all_results


def train_lgbm_quick():
    """Mode rapide sans Optuna (test)."""

    print("=" * 60)
    print("MODÈLE 2 : LightGBM — MODE RAPIDE (sans Optuna)")
    print("=" * 60)

    data = get_model_data(split_by_type=True)

    for type_name, d in data.items():
        max_flags = 7 if type_name == "K1" else 5
        num_classes = max_flags + 1

        model = lgb.LGBMClassifier(
            objective="multiclass", num_class=num_classes,
            n_estimators=150, learning_rate=0.05, num_leaves=24,
            max_depth=5, min_child_samples=15,
            subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.1, reg_lambda=0.1,
            random_state=42, verbosity=-1,
        )
        model.fit(d["train_X"], d["train_y"])

        test_pred = model.predict(d["test_X"])
        test_proba = model.predict_proba(d["test_X"])

        acc = accuracy_score(d["test_y"], test_pred)
        mae = mean_absolute_error(d["test_y"], test_pred)
        ll = log_loss(d["test_y"], test_proba, labels=list(range(num_classes)))
        acc_1 = np.mean(np.abs(d["test_y"] - test_pred) <= 1)

        print(f"\n  {type_name}: Acc={acc:.4f} | MAE={mae:.4f} | ±1={acc_1:.4f} | LL={ll:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Mode rapide sans Optuna")
    parser.add_argument("--trials", type=int, default=80, help="Nombre d'essais Optuna")
    args = parser.parse_args()

    if args.quick:
        train_lgbm_quick()
    else:
        train_lgbm_model(n_trials=args.trials)
