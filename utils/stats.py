"""Statistical testing utilities (normality, ANOVA, Chi², correlation)."""

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


def fmt_p(p_value: float) -> str:
    """Format a p-value to 3 decimals, with a ``< 0.001`` threshold."""
    if p_value < 0.0005:
        return "< 0.001"
    return f"{p_value:.3f}"


# ── Tests for a single numeric variable ──────────────────────────────────────

def normality_test_auto(sample: pd.Series) -> dict:
    sample = sample.dropna()
    n = len(sample)

    if n < 3:
        return {
            "test_name": "Normalité",
            "stat": np.nan,
            "p": np.nan,
            "comment": "Effectif insuffisant pour un test de normalité (n < 3).",
        }

    if n < 20:
        stat, p = sp_stats.shapiro(sample)
        test_name = "Shapiro-Wilk"
        comment = (
            "Test adapté aux petits échantillons (n < 20). "
            "Il est assez sensible aux déviations."
        )
    else:
        stat, p = sp_stats.normaltest(sample)
        test_name = "D'Agostino K²"
        comment = (
            "Test basé sur l'asymétrie et l'aplatissement, adapté aux échantillons ≥ 20. "
            "Avec des échantillons très grands, il peut détecter de très faibles déviations."
        )

    return {"test_name": test_name, "stat": stat, "p": p, "comment": comment}


# ── Y numeric ~ X categorical ────────────────────────────────────────────────

def oneway_test_auto(data: pd.DataFrame, y_col: str, x_group: str) -> dict:
    groups = []
    group_sizes = []

    for _, group in data.groupby(x_group):
        vals = group[y_col].dropna()
        n = len(vals)
        if n >= 3:
            groups.append(vals)
            group_sizes.append(n)

    if len(groups) < 2:
        return {
            "test_name": "Comparaison de moyennes",
            "stat": np.nan,
            "p": np.nan,
            "comment": "Pas assez de groupes avec au moins 3 observations.",
        }

    min_n = min(group_sizes)

    if min_n >= 30:
        stat, p = sp_stats.f_oneway(*groups)
        return {
            "test_name": "ANOVA (grand échantillon, TCL)",
            "stat": stat,
            "p": p,
            "comment": (
                "Tous les groupes ont n ≥ 30. ANOVA est robuste via le théorème central limite, "
                "même si la normalité n'est pas parfaite."
            ),
        }

    all_normal = True
    for vals, n in zip(groups, group_sizes):
        if n < 8:
            all_normal = False
        else:
            res = normality_test_auto(vals)
            if res["p"] is not None and not np.isnan(res["p"]) and res["p"] > 0.05:
                pass
            else:
                all_normal = False

    if all_normal:
        stat, p = sp_stats.f_oneway(*groups)
        return {
            "test_name": "ANOVA (normalité approximative)",
            "stat": stat,
            "p": p,
            "comment": (
                "Taille d'échantillon modérée et normalité non rejetée dans chaque groupe : "
                "ANOVA est appropriée."
            ),
        }
    else:
        stat, p = sp_stats.kruskal(*groups)
        return {
            "test_name": "Kruskal-Wallis",
            "stat": stat,
            "p": p,
            "comment": (
                "Normalité incertaine ou rejetée dans au moins un groupe et/ou effectif faible : "
                "on utilise le test non-paramétrique de Kruskal-Wallis."
            ),
        }


# ── X categorical ~ Y categorical ────────────────────────────────────────────

def chi2_or_fisher_auto(data: pd.DataFrame, x_cat: str, y_cat: str) -> dict:
    contingency = pd.crosstab(data[x_cat], data[y_cat])
    if contingency.empty:
        return {
            "test_name": "Test d'indépendance",
            "stat": np.nan,
            "p": np.nan,
            "comment": "Table de contingence vide, impossible de tester.",
            "dof": None,
        }

    chi2_stat, chi2_p, dof, expected = sp_stats.chi2_contingency(contingency)

    if contingency.shape == (2, 2) and (expected < 5).any():
        oddsratio, p = sp_stats.fisher_exact(contingency)
        return {
            "test_name": "Fisher exact (2x2, faibles effectifs)",
            "stat": oddsratio,
            "p": p,
            "comment": (
                "Tableau 2x2 avec au moins une fréquence attendue < 5 : "
                "le test exact de Fisher est plus approprié que le Chi²."
            ),
            "dof": None,
        }
    else:
        return {
            "test_name": "Chi² d'indépendance",
            "stat": chi2_stat,
            "p": chi2_p,
            "comment": "Conditions du Chi² vérifiées (fréquences attendues suffisantes).",
            "dof": dof,
        }


# ── Correlation (X numeric ~ Y numeric) ──────────────────────────────────────

def correlation_auto(x: pd.Series, y: pd.Series) -> dict:
    df = pd.concat([x, y], axis=1).dropna()
    if df.shape[0] < 3:
        return {
            "tests": [],
            "comment": "Effectif insuffisant pour un test de corrélation (n < 3).",
        }

    x_vals = df.iloc[:, 0]
    y_vals = df.iloc[:, 1]
    n = len(df)

    results = []

    if n < 20:
        res_x = normality_test_auto(x_vals)
        res_y = normality_test_auto(y_vals)
        x_normal = res_x["p"] is not None and not np.isnan(res_x["p"]) and res_x["p"] > 0.05
        y_normal = res_y["p"] is not None and not np.isnan(res_y["p"]) and res_y["p"] > 0.05

        if x_normal and y_normal:
            r, p = sp_stats.pearsonr(x_vals, y_vals)
            results.append({
                "name": "Pearson",
                "r": r,
                "p": p,
                "comment": (
                    "Petits effectifs (n < 20) et normalité non rejetée pour X et Y : "
                    "corrélation de Pearson appropriée."
                ),
            })
        else:
            rho, p = sp_stats.spearmanr(x_vals, y_vals)
            results.append({
                "name": "Spearman",
                "r": rho,
                "p": p,
                "comment": (
                    "Petits effectifs (n < 20) et normalité incertaine/rejetée : "
                    "corrélation de Spearman (non-paramétrique) préférable."
                ),
            })
    else:
        r_pearson, p_pearson = sp_stats.pearsonr(x_vals, y_vals)
        results.append({
            "name": "Pearson",
            "r": r_pearson,
            "p": p_pearson,
            "comment": (
                "Effectif n ≥ 20 : la corrélation de Pearson est robuste même en cas de légère "
                "déviation à la normalité (TCL)."
            ),
        })

        rho_spearman, p_spearman = sp_stats.spearmanr(x_vals, y_vals)
        results.append({
            "name": "Spearman",
            "r": rho_spearman,
            "p": p_spearman,
            "comment": (
                "Spearman mesure la corrélation monotone basée sur les rangs, "
                "utile en cas de relations non linéaires ou de valeurs extrêmes."
            ),
        })

    return {"tests": results, "comment": None}
