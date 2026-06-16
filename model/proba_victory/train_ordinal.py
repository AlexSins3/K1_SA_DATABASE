"""
Modèle 1 : Régression Logistique Ordinale — V2 (séparé K1/SA)
===============================================================
Prédit le nombre de drapeaux de l'athlète A (variable ordinale).

Avantages :
- Interprétable (coefficients)
- Respecte la nature ordonnée de la target
- Modèles séparés K1 (0-7) et SA (0-5)
"""

import sys
from pathlib import Path
import pickle
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, log_loss, classification_report,
    confusion_matrix, mean_absolute_error
)

try:
    from mord import LogisticAT
    HAS_MORD = True
except ImportError:
    HAS_MORD = False
    print("⚠️  Package 'mord' non installé → pip install mord")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_data import get_model_data, FEATURE_COLS_SPLIT


OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)


def train_single_model(train_X, train_y, test_X, test_y, type_name: str):
    """Entraîne un modèle ordinal optimisé (alpha + feature selection via CV)."""

    print(f"\n{'═' * 50}")
    print(f"  {type_name} — Régression Logistique Ordinale (optimisée)")
    print(f"{'═' * 50}")

    all_features = list(train_X.columns)

    # ──────────────────────────────────────────────────
    # Phase 1 : Recherche du meilleur alpha par CV
    # ──────────────────────────────────────────────────
    alphas = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    best_alpha = 1.0
    best_cv_score = -1

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n  🔍 Recherche alpha (CV 5-fold, métrique=Acc±1) :")
    for alpha in alphas:
        scores = []
        for tr_idx, va_idx in skf.split(train_X, train_y):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(train_X.iloc[tr_idx])
            X_va = scaler.transform(train_X.iloc[va_idx])
            y_tr = train_y[tr_idx]
            y_va = train_y[va_idx]

            if HAS_MORD:
                m = LogisticAT(alpha=alpha)
            else:
                m = LogisticRegression(
                    multi_class="multinomial", solver="lbfgs",
                    max_iter=2000, C=1.0/alpha, random_state=42,
                )
            m.fit(X_tr, y_tr)
            pred = m.predict(X_va)
            scores.append(np.mean(np.abs(y_va - pred) <= 1))

        mean_score = np.mean(scores)
        if mean_score > best_cv_score:
            best_cv_score = mean_score
            best_alpha = alpha

    print(f"     → Meilleur alpha={best_alpha}, Acc±1 CV={best_cv_score:.4f}")

    # ──────────────────────────────────────────────────
    # Phase 2 : Feature selection (backward elimination)
    # ──────────────────────────────────────────────────
    print(f"\n  🔍 Feature selection (backward, {len(all_features)} features) :")
    current_features = list(all_features)
    current_best = best_cv_score

    # Score de base avec toutes les features
    while len(current_features) > 5:
        worst_feat = None
        best_score_without = -1

        for feat in current_features:
            test_feats = [f for f in current_features if f != feat]
            scores = []
            for tr_idx, va_idx in skf.split(train_X, train_y):
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(train_X[test_feats].iloc[tr_idx])
                X_va = scaler.transform(train_X[test_feats].iloc[va_idx])
                y_tr = train_y[tr_idx]
                y_va = train_y[va_idx]

                if HAS_MORD:
                    m = LogisticAT(alpha=best_alpha)
                else:
                    m = LogisticRegression(
                        multi_class="multinomial", solver="lbfgs",
                        max_iter=2000, C=1.0/best_alpha, random_state=42,
                    )
                m.fit(X_tr, y_tr)
                pred = m.predict(X_va)
                scores.append(np.mean(np.abs(y_va - pred) <= 1))

            mean_s = np.mean(scores)
            if mean_s > best_score_without:
                best_score_without = mean_s
                worst_feat = feat

        # Si retirer une feature améliore ou ne dégrade pas, on la retire
        if best_score_without >= current_best - 0.005:
            current_features.remove(worst_feat)
            current_best = best_score_without
            print(f"     − Retiré '{worst_feat}' → Acc±1 CV={current_best:.4f} ({len(current_features)} features)")
        else:
            break

    selected_features = current_features
    print(f"     → {len(selected_features)} features retenues")

    # ──────────────────────────────────────────────────
    # Phase 3 : Entraînement final
    # ──────────────────────────────────────────────────
    scaler = StandardScaler()
    train_X_s = scaler.fit_transform(train_X[selected_features])
    test_X_s = scaler.transform(test_X[selected_features])

    if HAS_MORD:
        model = LogisticAT(alpha=best_alpha)
        model.fit(train_X_s, train_y)
        test_pred = model.predict(test_X_s)
        test_proba = None
    else:
        model = LogisticRegression(
            multi_class="multinomial", solver="lbfgs",
            max_iter=2000, C=1.0/best_alpha, random_state=42,
        )
        model.fit(train_X_s, train_y)
        test_pred = model.predict(test_X_s)
        test_proba = model.predict_proba(test_X_s)

    # Métriques
    acc = accuracy_score(test_y, test_pred)
    mae = mean_absolute_error(test_y, test_pred)
    acc_1 = np.mean(np.abs(test_y - test_pred) <= 1)
    ll = None
    if test_proba is not None:
        ll = log_loss(test_y, test_proba, labels=model.classes_)

    print(f"\n  📊 Accuracy exacte:  {acc:.4f}")
    print(f"  📊 MAE (drapeaux):   {mae:.4f}")
    print(f"  📊 Accuracy ±1:      {acc_1:.4f}")
    if ll is not None:
        print(f"  📊 Log-loss:         {ll:.4f}")

    print(f"\n  📋 Classification Report:")
    print(classification_report(test_y, test_pred, zero_division=0))

    # Coefficients
    if hasattr(model, "coef_"):
        coefs = model.coef_ if HAS_MORD else np.mean(np.abs(model.coef_), axis=0)
        feat_label = "Coefficient" if HAS_MORD else "|Coef| moyen"
        feat_df = pd.DataFrame({
            "Feature": selected_features,
            feat_label: coefs
        }).sort_values(feat_label, key=abs, ascending=False)
        print(f"\n  📋 Top features ({feat_label}) :")
        print(feat_df.head(10).to_string(index=False))

    return {
        "model": model, "scaler": scaler,
        "selected_features": selected_features,
        "metrics": {"accuracy": float(acc), "mae": float(mae),
                    "accuracy_pm1": float(acc_1), "log_loss": float(ll) if ll else None,
                    "best_alpha": float(best_alpha), "n_features": len(selected_features)},
    }


def train_ordinal_model():
    """Entraîne les modèles ordinaux séparés K1 et SA."""

    print("=" * 60)
    print("MODÈLE 1 : RÉGRESSION LOGISTIQUE ORDINALE — V2")
    print("  (modèles séparés K1 / SA)")
    print("=" * 60)

    data = get_model_data(split_by_type=True)

    all_results = {}
    for type_name, d in data.items():
        result = train_single_model(
            d["train_X"], d["train_y"], d["test_X"], d["test_y"], type_name
        )
        all_results[type_name] = result

    # Sauvegarde
    save_data = {}
    save_metrics = {}
    for type_name, r in all_results.items():
        save_data[type_name] = {
            "model": r["model"], "scaler": r["scaler"],
            "selected_features": r["selected_features"],
        }
        save_metrics[type_name] = r["metrics"]

    with open(OUTPUT_DIR / "ordinal_models.pkl", "wb") as f:
        pickle.dump(save_data, f)

    with open(OUTPUT_DIR / "ordinal_results.json", "w") as f:
        json.dump(save_metrics, f, indent=2, default=str)

    print(f"\n✅ Modèles sauvegardés dans {OUTPUT_DIR}/")
    return all_results


if __name__ == "__main__":
    train_ordinal_model()
