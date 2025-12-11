# tabs/graphs.py

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy import stats


# =========================
# Helpers génériques
# =========================

def _get_numeric_and_categorical_columns(df: pd.DataFrame):
    numeric_cols = [
        col for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].dtype != bool
    ]
    categorical_cols = [col for col in df.columns if col not in numeric_cols]
    return numeric_cols, categorical_cols


def _fmt_p(p_value: float) -> str:
    """Formatage standardisé des p-values à 3 décimales, avec seuil."""
    if p_value < 0.0005:
        return "< 0.001"
    return f"{p_value:.3f}"


# =========================
# Tests pour 1 variable numérique
# =========================

def normality_test_auto(sample: pd.Series):
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
        stat, p = stats.shapiro(sample)
        test_name = "Shapiro-Wilk"
        comment = (
            "Test adapté aux petits échantillons (n < 20). "
            "Il est assez sensible aux déviations."
        )
    else:
        stat, p = stats.normaltest(sample)
        test_name = "D'Agostino K²"
        comment = (
            "Test basé sur l'asymétrie et l'aplatissement, adapté aux échantillons ≥ 20. "
            "Avec des échantillons très grands, il peut détecter de très faibles déviations."
        )

    return {
        "test_name": test_name,
        "stat": stat,
        "p": p,
        "comment": comment,
    }


# =========================
# Tests Y numérique ~ X catégorielle
# =========================

def oneway_test_auto(data: pd.DataFrame, y_col: str, x_group: str):
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
        stat, p = stats.f_oneway(*groups)
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
        stat, p = stats.f_oneway(*groups)
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
        stat, p = stats.kruskal(*groups)
        return {
            "test_name": "Kruskal-Wallis",
            "stat": stat,
            "p": p,
            "comment": (
                "Normalité incertaine ou rejetée dans au moins un groupe et/ou effectif faible : "
                "on utilise le test non-paramétrique de Kruskal-Wallis."
            ),
        }


# =========================
# Tests X catégorielle ~ Y catégorielle
# =========================

def chi2_or_fisher_auto(data: pd.DataFrame, x_cat: str, y_cat: str):
    contingency = pd.crosstab(data[x_cat], data[y_cat])
    if contingency.empty:
        return {
            "test_name": "Test d'indépendance",
            "stat": np.nan,
            "p": np.nan,
            "comment": "Table de contingence vide, impossible de tester.",
            "dof": None,
        }

    chi2_stat, chi2_p, dof, expected = stats.chi2_contingency(contingency)

    if contingency.shape == (2, 2) and (expected < 5).any():
        oddsratio, p = stats.fisher_exact(contingency)
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
            "comment": (
                "Conditions du Chi² vérifiées (fréquences attendues suffisantes)."
            ),
            "dof": dof,
        }


# =========================
# Corrélation X numérique ~ Y numérique
# =========================

def correlation_auto(x: pd.Series, y: pd.Series):
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
            r, p = stats.pearsonr(x_vals, y_vals)
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
            rho, p = stats.spearmanr(x_vals, y_vals)
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
        r_pearson, p_pearson = stats.pearsonr(x_vals, y_vals)
        results.append({
            "name": "Pearson",
            "r": r_pearson,
            "p": p_pearson,
            "comment": (
                "Effectif n ≥ 20 : la corrélation de Pearson est robuste même en cas de légère "
                "déviation à la normalité (TCL)."
            ),
        })

        rho_spearman, p_spearman = stats.spearmanr(x_vals, y_vals)
        results.append({
            "name": "Spearman",
            "r": rho_spearman,
            "p": p_spearman,
            "comment": (
                "Spearman mesure la corrélation monotone basée sur les rangs, "
                "utile en cas de relations non linéaires ou de valeurs extrêmes."
            ),
        })

    return {
        "tests": results,
        "comment": None,
    }


# =========================
# Onglet principal
# =========================

def show_graphs_tab(data: pd.DataFrame) -> None:
    st.header("Générateur de Graphiques Interactifs")

    df = data.copy()

    # Mise en page : bande de filtres à gauche, contenu à droite
    filters_col, content_col = st.columns([0.9, 2.5])

    # Variables partagées
    data_compet = df.copy()
    x_num = "Aucune"
    x_cat = "Aucune"
    y_num = "Aucune"
    y_cat = "Aucune"
    filtre_var = "Aucune"
    modalites_selectionnees = None

    # =========================
    # Colonne gauche : filtres
    # =========================
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
            .filter-panel label {
                font-weight: 500;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div class='filter-panel'>", unsafe_allow_html=True)
        st.markdown("### 🎛️ Filtres")

        # Type de compétition
        type_compet_options = ["Tous", "Premier League (K1)", "Series A (SA)"]
        selected_type_compet = st.radio(
            "Type de compétition",
            type_compet_options,
            key="graphs_type_compet",
        )

        if selected_type_compet != "Tous":
            if selected_type_compet == "Premier League (K1)":
                data_compet = df[df['Type_Compet'] == 'K1'].copy()
            elif selected_type_compet == "Series A (SA)":
                data_compet = df[df['Type_Compet'] == 'SA'].copy()
        else:
            data_compet = df.copy()

        variables_numeriques, variables_categorielles = _get_numeric_and_categorical_columns(data_compet)

        st.markdown("---")
        st.markdown("#### Axes du graphique")

        # Axe X
        x_num = st.selectbox(
            "Variable numérique (X)",
            ["Aucune"] + variables_numeriques,
            key="graphs_x_num",
        )
        x_cat = st.selectbox(
            "Variable catégorielle (X)",
            ["Aucune"] + variables_categorielles,
            key="graphs_x_cat",
        )

        # Axe Y
        y_num = st.selectbox(
            "Variable numérique (Y)",
            ["Aucune"] + variables_numeriques,
            key="graphs_y_num",
        )
        y_cat = st.selectbox(
            "Variable catégorielle (Y)",
            ["Aucune"] + variables_categorielles,
            key="graphs_y_cat",
        )

        # Filtres supplémentaires par cas (on choisira case par case côté droit)
        st.markdown("</div>", unsafe_allow_html=True)

    # =========================
    # Colonne droite : texte + graphes + tests
    # =========================
    with content_col:
        st.markdown(
            """
            ### Comment utiliser les graphiques interactifs

            1. Choisissez **une variable Y** (numérique ou catégorielle) et éventuellement **une variable X**.  
            2. La combinaison X/Y choisie déterminera automatiquement :
               - Le type de graphique (histogramme, boxplot, barres empilées, nuage de points…)
               - Le **test statistique le plus adapté** (normalité, ANOVA, Kruskal, Chi², Fisher, corrélation…)
            3. Le test et la p-value sont affichés **sous le graphique**.
            """
        )

        st.subheader("Type de graphique généré")

        # =========
        # Cas 1 : une seule variable numérique en Y
        # =========
        if y_num != "Aucune" and x_num == "Aucune" and x_cat == "Aucune" and y_cat == "Aucune":
            st.markdown(f"**Distribution de la variable `{y_num}`**")

            # Filtres pour ce cas
            with filters_col:
                st.markdown("<div class='filter-panel'>", unsafe_allow_html=True)
                st.markdown("#### Filtre (optionnel)")
                filtre_var = st.selectbox(
                    "Variable catégorielle pour filtrer",
                    ["Aucune"] + variables_categorielles,
                    key="graphs_filter_num_only",
                )

                if filtre_var != "Aucune":
                    modalites = data_compet[filtre_var].dropna().unique().tolist()
                    modalites_selectionnees = st.multiselect(
                        f"Modalités de {filtre_var}",
                        modalites,
                        default=modalites,
                        key="graphs_modalites_num_only",
                    )
                st.markdown("</div>", unsafe_allow_html=True)

            # Construction des données filtrées
            if filtre_var != "Aucune" and modalites_selectionnees is not None:
                data_filtered = data_compet[data_compet[filtre_var].isin(modalites_selectionnees)].copy()
            else:
                data_filtered = data_compet.copy()

            data_filtered = data_filtered[data_filtered[y_num].notna()]

            if data_filtered.empty:
                st.warning("Aucune donnée disponible avec ces filtres.")
                return

            fig = px.histogram(data_filtered, x=y_num, nbins=30, marginal='box')

            mean_value = data_filtered[y_num].mean()
            median_value = data_filtered[y_num].median()

            fig.add_vline(
                x=mean_value,
                line_dash='dash',
                line_color='red',
                annotation_text='Moyenne',
                annotation_position='top left',
            )
            fig.add_vline(
                x=median_value,
                line_dash='dash',
                line_color='green',
                annotation_text='Médiane',
                annotation_position='top right',
            )

            st.plotly_chart(fig, width="stretch")

            st.subheader("Test de normalité (automatique)")
            if st.button("Effectuer le test de normalité", key="graphs_btn_normality"):
                res = normality_test_auto(data_filtered[y_num])

                if np.isnan(res["stat"]) or np.isnan(res["p"]):
                    st.warning(res["comment"])
                else:
                    st.write(
                        f"Test utilisé : **{res['test_name']}**  \n"
                        f"- Statistique : **{res['stat']:.3f}**  \n"
                        f"- p-value : **{_fmt_p(res['p'])}**"
                    )

                    if res["p"] > 0.05:
                        interpretation = (
                            f"La normalité **n'est pas rejetée** pour **{y_num}** (p = {_fmt_p(res['p'])})."
                        )
                    else:
                        interpretation = (
                            f"La normalité est **rejetée** pour **{y_num}** (p = {_fmt_p(res['p'])})."
                        )

                    st.write(interpretation)
                    st.markdown(res["comment"])

        # =========
        # Cas 2 : Y numérique, X catégorielle
        # =========
        elif y_num != "Aucune" and x_cat != "Aucune" and x_num == "Aucune" and y_cat == "Aucune":
            st.markdown(f"**Distribution de `{y_num}` par rapport à chaque modalité de `{x_cat}`**")

            with filters_col:
                st.markdown("<div class='filter-panel'>", unsafe_allow_html=True)
                st.markdown("#### Filtre (optionnel)")
                filtre_var = st.selectbox(
                    "Variable catégorielle pour filtrer",
                    ["Aucune"] + variables_categorielles,
                    key="graphs_filter_y_num_x_cat",
                )

                if filtre_var != "Aucune":
                    modalites = data_compet[filtre_var].dropna().unique().tolist()
                    modalites_selectionnees = st.multiselect(
                        f"Modalités de {filtre_var}",
                        modalites,
                        default=modalites,
                        key="graphs_modalites_y_num_x_cat",
                    )
                st.markdown("</div>", unsafe_allow_html=True)

            if filtre_var != "Aucune" and modalites_selectionnees is not None:
                data_filtered = data_compet[data_compet[filtre_var].isin(modalites_selectionnees)].copy()
            else:
                data_filtered = data_compet.copy()

            data_filtered = data_filtered[data_filtered[y_num].notna()]

            if data_filtered.empty:
                st.warning("Aucune donnée disponible avec ces filtres.")
                return

            fig = px.box(data_filtered, x=x_cat, y=y_num, points='all')
            st.plotly_chart(fig, width="stretch")

            st.subheader("Test Statistique (automatique)")
            if st.button("Effectuer le test statistique", key="graphs_btn_y_num_x_cat"):
                res = oneway_test_auto(data_filtered, y_num, x_cat)

                if np.isnan(res["stat"]) or np.isnan(res["p"]):
                    st.warning(res["comment"])
                else:
                    st.write(
                        f"Test utilisé : **{res['test_name']}**  \n"
                        f"- Statistique : **{res['stat']:.3f}**  \n"
                        f"- p-value : **{_fmt_p(res['p'])}**"
                    )

                    if res["p"] < 0.05:
                        conclusion = (
                            f"Différence **significative** de **{y_num}** entre les groupes de **{x_cat}** "
                            f"(p = {_fmt_p(res['p'])})."
                        )
                    else:
                        conclusion = (
                            f"Aucune différence significative de **{y_num}** entre les groupes de **{x_cat}** "
                            f"(p = {_fmt_p(res['p'])})."
                        )

                    st.write(conclusion)
                    st.markdown(res["comment"])

        # =========
        # Cas 3 : une seule variable catégorielle en Y
        # =========
        elif y_cat != "Aucune" and x_num == "Aucune" and x_cat == "Aucune" and y_num == "Aucune":
            with filters_col:
                st.markdown("<div class='filter-panel'>", unsafe_allow_html=True)
                st.markdown("#### Filtre (optionnel)")
                filtre_var = st.selectbox(
                    "Variable catégorielle pour filtrer",
                    ["Aucune"] + variables_categorielles,
                    key="graphs_filter_y_cat_only",
                )

                if filtre_var != "Aucune":
                    modalites = data_compet[filtre_var].dropna().unique().tolist()
                    modalites_selectionnees = st.multiselect(
                        f"Modalités de {filtre_var}",
                        modalites,
                        default=modalites,
                        key="graphs_modalites_y_cat_only",
                    )
                st.markdown("</div>", unsafe_allow_html=True)

            if filtre_var != "Aucune" and modalites_selectionnees is not None:
                data_filtered = data_compet[data_compet[filtre_var].isin(modalites_selectionnees)].copy()
            else:
                data_filtered = data_compet.copy()

            st.markdown(f"**Histogramme des effectifs de chaque modalité de `{y_cat}`**")

            counts = data_filtered[y_cat].value_counts().reset_index()
            counts.columns = [y_cat, 'Effectif']
            fig_count = px.bar(counts, x=y_cat, y='Effectif')
            st.plotly_chart(fig_count, width="stretch")

            st.markdown(f"**Histogramme des proportions de chaque modalité de `{y_cat}`**")
            counts_prop = data_filtered[y_cat].value_counts(normalize=True).reset_index()
            counts_prop.columns = [y_cat, 'Proportion']
            fig_prop = px.bar(counts_prop, x=y_cat, y='Proportion')
            st.plotly_chart(fig_prop, width="stretch")

        # =========
        # Cas 4 : X et Y catégorielles
        # =========
        elif y_cat != "Aucune" and x_cat != "Aucune" and y_num == "Aucune" and x_num == "Aucune":
            st.markdown(f"**Proportions des modalités de `{y_cat}` en fonction de `{x_cat}`**")

            with filters_col:
                st.markdown("<div class='filter-panel'>", unsafe_allow_html=True)
                st.markdown("#### Filtre (optionnel)")
                filtre_var = st.selectbox(
                    "Variable catégorielle pour filtrer",
                    ["Aucune"] + variables_categorielles,
                    key="graphs_filter_x_y_cat",
                )

                if filtre_var != "Aucune":
                    modalites = data_compet[filtre_var].dropna().unique().tolist()
                    modalites_selectionnees = st.multiselect(
                        f"Modalités de {filtre_var}",
                        modalites,
                        default=modalites,
                        key="graphs_modalites_x_y_cat",
                    )
                st.markdown("</div>", unsafe_allow_html=True)

            if filtre_var != "Aucune" and modalites_selectionnees is not None:
                data_filtered = data_compet[data_compet[filtre_var].isin(modalites_selectionnees)].copy()
            else:
                data_filtered = data_compet.copy()

            if data_filtered.empty:
                st.warning("Aucune donnée disponible avec ces filtres.")
                return

            crosstab = pd.crosstab(data_filtered[x_cat], data_filtered[y_cat], normalize='index')
            crosstab.reset_index(inplace=True)
            crosstab_melted = crosstab.melt(id_vars=x_cat, var_name=y_cat, value_name='Proportion')

            fig = px.bar(
                crosstab_melted,
                x=x_cat,
                y='Proportion',
                color=y_cat,
                barmode='stack',
            )
            st.plotly_chart(fig, width="stretch")

            st.subheader("Test Statistique (automatique)")
            if st.button("Effectuer le test d'indépendance", key="graphs_btn_x_y_cat"):
                res = chi2_or_fisher_auto(data_filtered, x_cat, y_cat)

                if np.isnan(res["stat"]) or np.isnan(res["p"]):
                    st.warning(res["comment"])
                else:
                    st.write(
                        f"Test utilisé : **{res['test_name']}**  \n"
                        f"- Statistique : **{res['stat']:.3f}**"
                    )
                    if res["p"] is not None:
                        st.write(f"- p-value : **{_fmt_p(res['p'])}**")
                    if res["dof"] is not None:
                        st.write(f"- Degrés de liberté : **{res['dof']}**")

                    if res["p"] is not None:
                        if res["p"] < 0.05:
                            conclusion = (
                                f"Association **significative** entre **{x_cat}** et **{y_cat}** "
                                f"(p = {_fmt_p(res['p'])})."
                            )
                        else:
                            conclusion = (
                                f"Aucune association significative entre **{x_cat}** et **{y_cat}** "
                                f"(p = {_fmt_p(res['p'])})."
                            )
                        st.write(conclusion)

                    st.markdown(res["comment"])

        # =========
        # Cas 5 : X et Y numériques
        # =========
        elif x_num != "Aucune" and y_num != "Aucune" and x_cat == "Aucune" and y_cat == "Aucune":
            st.markdown(f"**Nuage de points entre `{x_num}` et `{y_num}`**")

            with filters_col:
                st.markdown("<div class='filter-panel'>", unsafe_allow_html=True)
                st.markdown("#### Filtre (optionnel)")
                filtre_var = st.selectbox(
                    "Variable catégorielle pour filtrer",
                    ["Aucune"] + variables_categorielles,
                    key="graphs_filter_x_y_num",
                )

                if filtre_var != "Aucune":
                    modalites = data_compet[filtre_var].dropna().unique().tolist()
                    modalites_selectionnees = st.multiselect(
                        f"Modalités de {filtre_var}",
                        modalites,
                        default=modalites,
                        key="graphs_modalites_x_y_num",
                    )
                st.markdown("</div>", unsafe_allow_html=True)

            if filtre_var != "Aucune" and modalites_selectionnees is not None:
                data_filtered = data_compet[data_compet[filtre_var].isin(modalites_selectionnees)].copy()
            else:
                data_filtered = data_compet.copy()

            data_filtered = data_filtered[data_filtered[x_num].notna() & data_filtered[y_num].notna()]

            if data_filtered.empty:
                st.warning("Aucune donnée disponible avec ces filtres.")
                return

            fig = px.scatter(data_filtered, x=x_num, y=y_num)
            st.plotly_chart(fig, width="stretch")

            st.subheader("Test de corrélation (automatique)")
            if st.button("Effectuer le test de corrélation", key="graphs_btn_corr"):
                res = correlation_auto(data_filtered[x_num], data_filtered[y_num])

                if not res["tests"]:
                    st.warning(res["comment"])
                else:
                    for test in res["tests"]:
                        st.write(
                            f"Test : **{test['name']}**  \n"
                            f"- Coefficient : **{test['r']:.3f}**  \n"
                            f"- p-value : **{_fmt_p(test['p'])}**"
                        )
                        if test["p"] < 0.05:
                            conclusion = (
                                f"Corrélation **significative** (p = {_fmt_p(test['p'])})."
                            )
                        else:
                            conclusion = (
                                f"Pas de corrélation significative (p = {_fmt_p(test['p'])})."
                            )
                        st.write(conclusion)
                        st.markdown(test["comment"])

        else:
            st.warning("Veuillez sélectionner des combinaisons cohérentes de variables pour générer un graphique.")
