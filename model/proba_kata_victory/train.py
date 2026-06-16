"""
Entraînement & Comparaison — Modèle de victoire par kata
=========================================================

Compare plusieurs modèles :
1. Logistic Regression (baseline, comme l'actuel)
2. LightGBM
3. Logistic Regression avec feature selection optimisée
4. LightGBM avec Optuna

Métrique principale : AUC-ROC (classification binaire)
Métriques secondaires : Accuracy, Brier score, Log-loss
"""

import sys
import warnings
from pathlib import Path
import pickle
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, roc_auc_score, log_loss, brier_score_loss,
    classification_report
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_data import get_model_data, ALL_FEATURES, FEATURES_NO_TRANSFER

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOGISTIC REGRESSION (BASELINE)
# ══════════════════════════════════════════════════════════════════════════════

def train_logistic(train_X, train_y, test_X, test_y, C=0.5):
    """Logistic Regression avec StandardScaler."""
    print(f"\n{'═' * 55}")
    print(f"  LOGISTIC REGRESSION (C={C})")
    print(f"{'═' * 55}")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, C=C, solver="lbfgs", class_weight="balanced")),
    ])
    pipe.fit(train_X, train_y)

    test_pred = pipe.predict(test_X)
    test_proba = pipe.predict_proba(test_X)[:, 1]

    metrics = _compute_metrics(test_y, test_pred, test_proba, "Logistic")
    _print_metrics(metrics)

    # Feature importance
    coefs = pipe.named_steps["clf"].coef_[0]
    feat_df = pd.DataFrame({
        "Feature": train_X.columns.tolist(),
        "Coefficient": coefs,
    }).sort_values("Coefficient", key=abs, ascending=False)
    print(f"\n  📋 Top 10 features:")
    print(feat_df.head(10).to_string(index=False))

    return pipe, metrics


# ══════════════════════════════════════════════════════════════════════════════
# 2. LOGISTIC + FEATURE SELECTION (CV)
# ══════════════════════════════════════════════════════════════════════════════

def train_logistic_optimized(train_X, train_y, test_X, test_y):
    """Logistic Regression avec search de C + backward feature selection."""
    print(f"\n{'═' * 55}")
    print(f"  LOGISTIC OPTIMISÉE (C search + feature selection)")
    print(f"{'═' * 55}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    features = list(train_X.columns)

    # Phase 1: Best C
    best_c = 0.5
    best_auc = 0.0
    for c in [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=c, solver="lbfgs", class_weight="balanced")),
        ])
        scores = cross_val_score(pipe, train_X, train_y, cv=skf, scoring="roc_auc")
        if scores.mean() > best_auc:
            best_auc = scores.mean()
            best_c = c

    print(f"  → Best C={best_c}, AUC CV={best_auc:.4f}")

    # Phase 2: Backward elimination
    current_features = list(features)
    current_best = best_auc

    print(f"  🔍 Feature selection (backward, {len(current_features)} features):")
    while len(current_features) > 8:
        worst_feat = None
        best_without = -1

        for feat in current_features:
            test_feats = [f for f in current_features if f != feat]
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, C=best_c, solver="lbfgs", class_weight="balanced")),
            ])
            scores = cross_val_score(pipe, train_X[test_feats], train_y, cv=skf, scoring="roc_auc")
            if scores.mean() > best_without:
                best_without = scores.mean()
                worst_feat = feat

        if best_without >= current_best - 0.003:
            current_features.remove(worst_feat)
            current_best = best_without
            print(f"     − Retiré '{worst_feat}' → AUC CV={current_best:.4f} ({len(current_features)} feat)")
        else:
            break

    print(f"  → {len(current_features)} features retenues, AUC CV={current_best:.4f}")

    # Final training
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, C=best_c, solver="lbfgs", class_weight="balanced")),
    ])
    pipe.fit(train_X[current_features], train_y)

    test_pred = pipe.predict(test_X[current_features])
    test_proba = pipe.predict_proba(test_X[current_features])[:, 1]

    metrics = _compute_metrics(test_y, test_pred, test_proba, "Logistic_Opt")
    metrics["n_features"] = len(current_features)
    metrics["best_C"] = best_c
    _print_metrics(metrics)

    return pipe, metrics, current_features


# ══════════════════════════════════════════════════════════════════════════════
# 3. LIGHTGBM
# ══════════════════════════════════════════════════════════════════════════════

def train_lgbm_default(train_X, train_y, test_X, test_y):
    """LightGBM avec hyperparamètres raisonnables."""
    if not HAS_LGBM:
        print("\n  ⚠️ LightGBM non installé, skip")
        return None, None

    print(f"\n{'═' * 55}")
    print(f"  LIGHTGBM (default params)")
    print(f"{'═' * 55}")

    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=32,
        max_depth=5,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=42,
        verbosity=-1,
    )
    model.fit(train_X, train_y)

    test_pred = model.predict(test_X)
    test_proba = model.predict_proba(test_X)[:, 1]
    train_proba = model.predict_proba(train_X)[:, 1]
    train_auc = roc_auc_score(train_y, train_proba)

    metrics = _compute_metrics(test_y, test_pred, test_proba, "LightGBM")
    metrics["train_auc"] = train_auc
    _print_metrics(metrics)
    print(f"  📊 Train AUC: {train_auc:.4f} (overfit check)")

    # Feature importance
    fi = pd.DataFrame({
        "Feature": train_X.columns.tolist(),
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False)
    print(f"\n  📋 Top 10 features:")
    print(fi.head(10).to_string(index=False))

    return model, metrics


# ══════════════════════════════════════════════════════════════════════════════
# 4. LIGHTGBM + OPTUNA
# ══════════════════════════════════════════════════════════════════════════════

def train_lgbm_optuna(train_X, train_y, test_X, test_y, n_trials=60):
    """LightGBM optimisé par Optuna."""
    if not HAS_LGBM or not HAS_OPTUNA:
        print("\n  ⚠️ LightGBM ou Optuna non installé, skip")
        return None, None

    print(f"\n{'═' * 55}")
    print(f"  LIGHTGBM + OPTUNA ({n_trials} trials)")
    print(f"{'═' * 55}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 64),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.001, 10.0, log=True),
        }
        model = lgb.LGBMClassifier(
            **params, class_weight="balanced", random_state=42, verbosity=-1,
        )
        scores = cross_val_score(model, train_X, train_y, cv=skf, scoring="roc_auc")
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"  → Best AUC CV: {study.best_value:.4f}")
    print(f"  → Best params: {study.best_params}")

    # Retrain with best params
    model = lgb.LGBMClassifier(
        **study.best_params, class_weight="balanced", random_state=42, verbosity=-1,
    )
    model.fit(train_X, train_y)

    test_pred = model.predict(test_X)
    test_proba = model.predict_proba(test_X)[:, 1]
    train_proba = model.predict_proba(train_X)[:, 1]
    train_auc = roc_auc_score(train_y, train_proba)

    metrics = _compute_metrics(test_y, test_pred, test_proba, "LightGBM_Optuna")
    metrics["train_auc"] = train_auc
    metrics["best_cv_auc"] = study.best_value
    _print_metrics(metrics)
    print(f"  📊 Train AUC: {train_auc:.4f} | CV AUC: {study.best_value:.4f}")

    return model, metrics


# ══════════════════════════════════════════════════════════════════════════════
# 5. GRADIENT BOOSTING (sklearn)
# ══════════════════════════════════════════════════════════════════════════════

def train_gradient_boosting(train_X, train_y, test_X, test_y):
    """GradientBoosting sklearn (plus régularisé)."""
    print(f"\n{'═' * 55}")
    print(f"  GRADIENT BOOSTING (sklearn)")
    print(f"{'═' * 55}")

    model = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=20,
        subsample=0.8,
        random_state=42,
    )
    model.fit(train_X, train_y)

    test_pred = model.predict(test_X)
    test_proba = model.predict_proba(test_X)[:, 1]
    train_proba = model.predict_proba(train_X)[:, 1]
    train_auc = roc_auc_score(train_y, train_proba)

    metrics = _compute_metrics(test_y, test_pred, test_proba, "GradientBoosting")
    metrics["train_auc"] = train_auc
    _print_metrics(metrics)
    print(f"  📊 Train AUC: {train_auc:.4f}")

    fi = pd.DataFrame({
        "Feature": train_X.columns.tolist(),
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False)
    print(f"\n  📋 Top 10 features:")
    print(fi.head(10).to_string(index=False))

    return model, metrics


# ══════════════════════════════════════════════════════════════════════════════
# UTILS
# ══════════════════════════════════════════════════════════════════════════════

def _compute_metrics(y_true, y_pred, y_proba, model_name):
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    ll = log_loss(y_true, y_proba)
    brier = brier_score_loss(y_true, y_proba)
    return {
        "model": model_name,
        "accuracy": float(acc),
        "auc_roc": float(auc),
        "log_loss": float(ll),
        "brier_score": float(brier),
    }


def _print_metrics(metrics):
    print(f"\n  📊 Accuracy:     {metrics['accuracy']:.4f}")
    print(f"  📊 AUC-ROC:      {metrics['auc_roc']:.4f}")
    print(f"  📊 Log-loss:     {metrics['log_loss']:.4f}")
    print(f"  📊 Brier score:  {metrics['brier_score']:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("MODÈLE DE VICTOIRE PAR KATA — COMPARAISON")
    print("=" * 60)

    train_X, train_y, test_X, test_y, full_df = get_model_data(ALL_FEATURES)

    all_results = {}

    # 1. Logistic baseline
    model_lr, metrics_lr = train_logistic(train_X, train_y, test_X, test_y, C=0.5)
    all_results["logistic"] = {"model": model_lr, "metrics": metrics_lr, "features": ALL_FEATURES}

    # 2. Logistic optimized
    model_lr_opt, metrics_lr_opt, selected_feats = train_logistic_optimized(train_X, train_y, test_X, test_y)
    all_results["logistic_opt"] = {"model": model_lr_opt, "metrics": metrics_lr_opt, "features": selected_feats}

    # 3. Gradient Boosting
    model_gb, metrics_gb = train_gradient_boosting(train_X, train_y, test_X, test_y)
    all_results["gradient_boosting"] = {"model": model_gb, "metrics": metrics_gb, "features": ALL_FEATURES}

    # 4. LightGBM default
    model_lgbm, metrics_lgbm = train_lgbm_default(train_X, train_y, test_X, test_y)
    if model_lgbm:
        all_results["lgbm"] = {"model": model_lgbm, "metrics": metrics_lgbm, "features": ALL_FEATURES}

    # 5. LightGBM Optuna
    model_lgbm_opt, metrics_lgbm_opt = train_lgbm_optuna(train_X, train_y, test_X, test_y, n_trials=60)
    if model_lgbm_opt:
        all_results["lgbm_optuna"] = {"model": model_lgbm_opt, "metrics": metrics_lgbm_opt, "features": ALL_FEATURES}

    # ── COMPARISON TABLE ──
    print(f"\n\n{'═' * 70}")
    print("  TABLEAU COMPARATIF")
    print(f"{'═' * 70}")
    print(f"\n  {'Modèle':<22} {'Accuracy':>10} {'AUC-ROC':>10} {'Log-loss':>10} {'Brier':>10}")
    print(f"  {'─' * 62}")

    best_model_name = None
    best_auc = 0.0

    for name, data in all_results.items():
        m = data["metrics"]
        if m is None:
            continue
        flag = " ★" if m["auc_roc"] > best_auc else ""
        if m["auc_roc"] > best_auc:
            best_auc = m["auc_roc"]
            best_model_name = name
        print(f"  {m['model']:<22} {m['accuracy']:>10.4f} {m['auc_roc']:>10.4f} {m['log_loss']:>10.4f} {m['brier_score']:>10.4f}{flag}")

    print(f"\n  🏆 Meilleur modèle (AUC-ROC) : {best_model_name}")

    # ── SAVE BEST ──
    best_data = all_results[best_model_name]
    save_payload = {
        "model": best_data["model"],
        "features": best_data["features"],
        "metrics": best_data["metrics"],
    }
    with open(OUTPUT_DIR / "best_model.pkl", "wb") as f:
        pickle.dump(save_payload, f)

    # Save all metrics
    all_metrics = {name: data["metrics"] for name, data in all_results.items() if data["metrics"]}
    with open(OUTPUT_DIR / "comparison_results.json", "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    print(f"\n✅ Meilleur modèle sauvegardé dans {OUTPUT_DIR}/best_model.pkl")
    print(f"   Comparaison sauvegardée dans {OUTPUT_DIR}/comparison_results.json")

    return all_results


if __name__ == "__main__":
    main()
