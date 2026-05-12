# tabs/graphs.py

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.ui import filter_panel_open, filter_panel_close
from utils.data_helpers import get_numeric_and_categorical_columns
from utils.stats import (
    fmt_p,
    normality_test_auto,
    oneway_test_auto,
    chi2_or_fisher_auto,
    correlation_auto,
)
from utils.interpretations import show_tab_help
from utils.display import fmt_col


# ═══════════════════════════════════════════════════════════════════════════════
# Main tab
# ═══════════════════════════════════════════════════════════════════════════════

@st.fragment
def show_graphs_tab(data: pd.DataFrame) -> None:
    st.header("Générateur de Graphiques Interactifs")
    show_tab_help("graphs")

    df = data.copy()

    filters_col, content_col = st.columns([0.9, 2.5])

    data_compet = df.copy()
    x_num = "Aucune"
    x_cat = "Aucune"
    y_num = "Aucune"
    y_cat = "Aucune"
    filtre_var = "Aucune"
    modalites_selectionnees = None

    # ── Filtre gauche ─────────────────────────────────────────────────────────
    with filters_col:
        filter_panel_open()
        st.markdown("### 🎛️ Filtres")

        type_compet_options = ["Tous", "Premier League (K1)", "Series A (SA)"]
        selected_type_compet = st.radio(
            "Type de compétition", type_compet_options, key="graphs_type_compet",
        )

        if selected_type_compet == "Premier League (K1)":
            data_compet = df[df["Type_Compet"] == "K1"].copy()
        elif selected_type_compet == "Series A (SA)":
            data_compet = df[df["Type_Compet"] == "SA"].copy()
        else:
            data_compet = df.copy()

        variables_numeriques, variables_categorielles = get_numeric_and_categorical_columns(data_compet)

        st.markdown("---")
        st.markdown("#### Axes du graphique")

        x_num = st.selectbox("Variable numérique (X)", ["Aucune"] + variables_numeriques, format_func=fmt_col, key="graphs_x_num")
        x_cat = st.selectbox("Variable catégorielle (X)", ["Aucune"] + variables_categorielles, format_func=fmt_col, key="graphs_x_cat")
        y_num = st.selectbox("Variable numérique (Y)", ["Aucune"] + variables_numeriques, format_func=fmt_col, key="graphs_y_num")
        y_cat = st.selectbox("Variable catégorielle (Y)", ["Aucune"] + variables_categorielles, format_func=fmt_col, key="graphs_y_cat")

        filter_panel_close()

    # ── Contenu droit ─────────────────────────────────────────────────────────
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

        # ─── Cas 1 : une seule variable numérique en Y ───────────────────────
        if y_num != "Aucune" and x_num == "Aucune" and x_cat == "Aucune" and y_cat == "Aucune":
            st.markdown(f"**Distribution de la variable `{fmt_col(y_num)}`**")

            with filters_col:
                filter_panel_open()
                st.markdown("#### Filtre (optionnel)")
                filtre_var = st.selectbox(
                    "Variable catégorielle pour filtrer",
                    ["Aucune"] + variables_categorielles, key="graphs_filter_num_only",
                )
                if filtre_var != "Aucune":
                    modalites = data_compet[filtre_var].dropna().unique().tolist()
                    modalites_selectionnees = st.multiselect(
                        f"Modalités de {filtre_var}", modalites, default=modalites, key="graphs_modalites_num_only",
                    )
                filter_panel_close()

            data_filtered = _apply_optional_filter(data_compet, filtre_var, modalites_selectionnees)
            data_filtered = data_filtered[data_filtered[y_num].notna()]

            if data_filtered.empty:
                st.warning("Aucune donnée disponible avec ces filtres.")
                return

            fig = px.histogram(data_filtered, x=y_num, nbins=30, marginal="box")
            fig.add_vline(x=data_filtered[y_num].mean(), line_dash="dash", line_color="red",
                          annotation_text="Moyenne", annotation_position="top left")
            fig.add_vline(x=data_filtered[y_num].median(), line_dash="dash", line_color="green",
                          annotation_text="Médiane", annotation_position="top right")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Test de normalité (automatique)")
            if st.button("Effectuer le test de normalité", key="graphs_btn_normality"):
                _display_normality(data_filtered[y_num], y_num)

        # ─── Cas 2 : Y numérique, X catégorielle ─────────────────────────────
        elif y_num != "Aucune" and x_cat != "Aucune" and x_num == "Aucune" and y_cat == "Aucune":
            st.markdown(f"**Distribution de `{fmt_col(y_num)}` par rapport à chaque modalité de `{fmt_col(x_cat)}`**")

            with filters_col:
                filter_panel_open()
                st.markdown("#### Filtre (optionnel)")
                filtre_var = st.selectbox(
                    "Variable catégorielle pour filtrer",
                    ["Aucune"] + variables_categorielles, key="graphs_filter_y_num_x_cat",
                )
                if filtre_var != "Aucune":
                    modalites = data_compet[filtre_var].dropna().unique().tolist()
                    modalites_selectionnees = st.multiselect(
                        f"Modalités de {filtre_var}", modalites, default=modalites, key="graphs_modalites_y_num_x_cat",
                    )
                filter_panel_close()

            data_filtered = _apply_optional_filter(data_compet, filtre_var, modalites_selectionnees)
            data_filtered = data_filtered[data_filtered[y_num].notna()]

            if data_filtered.empty:
                st.warning("Aucune donnée disponible avec ces filtres.")
                return

            fig = px.box(data_filtered, x=x_cat, y=y_num, points="all")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Test Statistique (automatique)")
            if st.button("Effectuer le test statistique", key="graphs_btn_y_num_x_cat"):
                _display_oneway(data_filtered, y_num, x_cat)

        # ─── Cas 3 : une seule variable catégorielle en Y ────────────────────
        elif y_cat != "Aucune" and x_num == "Aucune" and x_cat == "Aucune" and y_num == "Aucune":
            with filters_col:
                filter_panel_open()
                st.markdown("#### Filtre (optionnel)")
                filtre_var = st.selectbox(
                    "Variable catégorielle pour filtrer",
                    ["Aucune"] + variables_categorielles, key="graphs_filter_y_cat_only",
                )
                if filtre_var != "Aucune":
                    modalites = data_compet[filtre_var].dropna().unique().tolist()
                    modalites_selectionnees = st.multiselect(
                        f"Modalités de {filtre_var}", modalites, default=modalites, key="graphs_modalites_y_cat_only",
                    )
                filter_panel_close()

            data_filtered = _apply_optional_filter(data_compet, filtre_var, modalites_selectionnees)

            st.markdown(f"**Histogramme des effectifs de chaque modalité de `{fmt_col(y_cat)}`**")
            counts = data_filtered[y_cat].value_counts().reset_index()
            counts.columns = [y_cat, "Effectif"]
            st.plotly_chart(px.bar(counts, x=y_cat, y="Effectif"), use_container_width=True)

            st.markdown(f"**Histogramme des proportions de chaque modalité de `{fmt_col(y_cat)}`**")
            counts_prop = data_filtered[y_cat].value_counts(normalize=True).reset_index()
            counts_prop.columns = [y_cat, "Proportion"]
            st.plotly_chart(px.bar(counts_prop, x=y_cat, y="Proportion"), use_container_width=True)

        # ─── Cas 4 : X et Y catégorielles ────────────────────────────────────
        elif y_cat != "Aucune" and x_cat != "Aucune" and y_num == "Aucune" and x_num == "Aucune":
            st.markdown(f"**Proportions des modalités de `{fmt_col(y_cat)}` en fonction de `{fmt_col(x_cat)}`**")

            with filters_col:
                filter_panel_open()
                st.markdown("#### Filtre (optionnel)")
                filtre_var = st.selectbox(
                    "Variable catégorielle pour filtrer",
                    ["Aucune"] + variables_categorielles, key="graphs_filter_x_y_cat",
                )
                if filtre_var != "Aucune":
                    modalites = data_compet[filtre_var].dropna().unique().tolist()
                    modalites_selectionnees = st.multiselect(
                        f"Modalités de {filtre_var}", modalites, default=modalites, key="graphs_modalites_x_y_cat",
                    )
                filter_panel_close()

            data_filtered = _apply_optional_filter(data_compet, filtre_var, modalites_selectionnees)
            if data_filtered.empty:
                st.warning("Aucune donnée disponible avec ces filtres.")
                return

            crosstab = pd.crosstab(data_filtered[x_cat], data_filtered[y_cat], normalize="index")
            crosstab.reset_index(inplace=True)
            crosstab_melted = crosstab.melt(id_vars=x_cat, var_name=y_cat, value_name="Proportion")
            st.plotly_chart(px.bar(crosstab_melted, x=x_cat, y="Proportion", color=y_cat, barmode="stack"), use_container_width=True)

            st.subheader("Test Statistique (automatique)")
            if st.button("Effectuer le test d'indépendance", key="graphs_btn_x_y_cat"):
                _display_chi2(data_filtered, x_cat, y_cat)

        # ─── Cas 5 : X et Y numériques ───────────────────────────────────────
        elif x_num != "Aucune" and y_num != "Aucune" and x_cat == "Aucune" and y_cat == "Aucune":
            st.markdown(f"**Nuage de points entre `{fmt_col(x_num)}` et `{fmt_col(y_num)}`**")

            with filters_col:
                filter_panel_open()
                st.markdown("#### Filtre (optionnel)")
                filtre_var = st.selectbox(
                    "Variable catégorielle pour filtrer",
                    ["Aucune"] + variables_categorielles, key="graphs_filter_x_y_num",
                )
                if filtre_var != "Aucune":
                    modalites = data_compet[filtre_var].dropna().unique().tolist()
                    modalites_selectionnees = st.multiselect(
                        f"Modalités de {filtre_var}", modalites, default=modalites, key="graphs_modalites_x_y_num",
                    )
                filter_panel_close()

            data_filtered = _apply_optional_filter(data_compet, filtre_var, modalites_selectionnees)
            data_filtered = data_filtered[data_filtered[x_num].notna() & data_filtered[y_num].notna()]

            if data_filtered.empty:
                st.warning("Aucune donnée disponible avec ces filtres.")
                return

            st.plotly_chart(px.scatter(data_filtered, x=x_num, y=y_num), use_container_width=True)

            st.subheader("Test de corrélation (automatique)")
            if st.button("Effectuer le test de corrélation", key="graphs_btn_corr"):
                _display_correlation(data_filtered, x_num, y_num)

        else:
            st.warning("Veuillez sélectionner des combinaisons cohérentes de variables pour générer un graphique.")


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers (display wrappers)
# ═══════════════════════════════════════════════════════════════════════════════

def _apply_optional_filter(data_compet, filtre_var, modalites_selectionnees):
    if filtre_var != "Aucune" and modalites_selectionnees is not None:
        return data_compet[data_compet[filtre_var].isin(modalites_selectionnees)].copy()
    return data_compet.copy()


def _display_normality(series, var_name):
    res = normality_test_auto(series)
    if np.isnan(res["stat"]) or np.isnan(res["p"]):
        st.warning(res["comment"])
    else:
        st.write(
            f"Test utilisé : **{res['test_name']}**  \n"
            f"- Statistique : **{res['stat']:.3f}**  \n"
            f"- p-value : **{fmt_p(res['p'])}**"
        )
        if res["p"] > 0.05:
            st.write(f"La normalité **n'est pas rejetée** pour **{var_name}** (p = {fmt_p(res['p'])}).")
        else:
            st.write(f"La normalité est **rejetée** pour **{var_name}** (p = {fmt_p(res['p'])}).")
        st.markdown(res["comment"])


def _display_oneway(data_filtered, y_num, x_cat):
    res = oneway_test_auto(data_filtered, y_num, x_cat)
    if np.isnan(res["stat"]) or np.isnan(res["p"]):
        st.warning(res["comment"])
    else:
        st.write(
            f"Test utilisé : **{res['test_name']}**  \n"
            f"- Statistique : **{res['stat']:.3f}**  \n"
            f"- p-value : **{fmt_p(res['p'])}**"
        )
        if res["p"] < 0.05:
            st.write(f"Différence **significative** de **{y_num}** entre les groupes de **{x_cat}** (p = {fmt_p(res['p'])}).")
        else:
            st.write(f"Aucune différence significative de **{y_num}** entre les groupes de **{x_cat}** (p = {fmt_p(res['p'])}).")
        st.markdown(res["comment"])


def _display_chi2(data_filtered, x_cat, y_cat):
    res = chi2_or_fisher_auto(data_filtered, x_cat, y_cat)
    if np.isnan(res["stat"]) or np.isnan(res["p"]):
        st.warning(res["comment"])
    else:
        st.write(f"Test utilisé : **{res['test_name']}**  \n- Statistique : **{res['stat']:.3f}**")
        if res["p"] is not None:
            st.write(f"- p-value : **{fmt_p(res['p'])}**")
        if res["dof"] is not None:
            st.write(f"- Degrés de liberté : **{res['dof']}**")
        if res["p"] is not None:
            if res["p"] < 0.05:
                st.write(f"Association **significative** entre **{x_cat}** et **{y_cat}** (p = {fmt_p(res['p'])}).")
            else:
                st.write(f"Aucune association significative entre **{x_cat}** et **{y_cat}** (p = {fmt_p(res['p'])}).")
        st.markdown(res["comment"])


def _display_correlation(data_filtered, x_num, y_num):
    res = correlation_auto(data_filtered[x_num], data_filtered[y_num])
    if not res["tests"]:
        st.warning(res["comment"])
    else:
        for test in res["tests"]:
            st.write(
                f"Test : **{test['name']}**  \n"
                f"- Coefficient : **{test['r']:.3f}**  \n"
                f"- p-value : **{fmt_p(test['p'])}**"
            )
            if test["p"] < 0.05:
                st.write(f"Corrélation **significative** (p = {fmt_p(test['p'])}).")
            else:
                st.write(f"Pas de corrélation significative (p = {fmt_p(test['p'])}).")
            st.markdown(test["comment"])
