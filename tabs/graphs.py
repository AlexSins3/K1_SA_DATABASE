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
from utils.lang import t, get_lang


# ═══════════════════════════════════════════════════════════════════════════════
# Main tab
# ═══════════════════════════════════════════════════════════════════════════════

@st.fragment
def show_graphs_tab(data: pd.DataFrame) -> None:
    st.header(t("Générateur de Graphiques Interactifs"))
    show_tab_help("graphs")

    df = data.copy()

    filters_col, content_col = st.columns([0.9, 2.5])

    _NONE = t("Aucune")

    data_compet = df.copy()
    x_num = _NONE
    x_cat = _NONE
    y_num = _NONE
    y_cat = _NONE
    filtre_var = _NONE
    modalites_selectionnees = None

    # ── Filtre gauche ─────────────────────────────────────────────────────────
    with filters_col:
        filter_panel_open()
        st.markdown(t("### 🎛️ Filtres"))

        type_compet_options = [t("Tous"), "Premier League (K1)", "Series A (SA)"]
        selected_type_compet = st.radio(
            t("Type de compétition"), type_compet_options, key="graphs_type_compet",
        )

        if selected_type_compet == "Premier League (K1)":
            data_compet = df[df["Type_Compet"] == "K1"].copy()
        elif selected_type_compet == "Series A (SA)":
            data_compet = df[df["Type_Compet"] == "SA"].copy()
        else:
            data_compet = df.copy()

        # Year filter
        if "Year" in data_compet.columns:
            years_g = sorted(pd.to_numeric(data_compet["Year"], errors="coerce").dropna().unique().tolist())
            sel_years_g = st.multiselect(
                t("Année(s)"), [int(y) for y in years_g],
                default=[int(y) for y in years_g], key="graphs_years"
            )
            if sel_years_g:
                data_compet = data_compet[pd.to_numeric(data_compet["Year"], errors="coerce").isin([float(y) for y in sel_years_g])]

        variables_numeriques, variables_categorielles = get_numeric_and_categorical_columns(data_compet)

        st.markdown("---")
        st.markdown(t("#### Axes du graphique"))

        x_num = st.selectbox(t("Variable numérique (X)"), [t("Aucune")] + variables_numeriques, format_func=fmt_col, key="graphs_x_num")
        x_cat = st.selectbox(t("Variable catégorielle (X)"), [t("Aucune")] + variables_categorielles, format_func=fmt_col, key="graphs_x_cat")
        y_num = st.selectbox(t("Variable numérique (Y)"), [t("Aucune")] + variables_numeriques, format_func=fmt_col, key="graphs_y_num")
        y_cat = st.selectbox(t("Variable catégorielle (Y)"), [t("Aucune")] + variables_categorielles, format_func=fmt_col, key="graphs_y_cat")

        filter_panel_close()

    # ── Contenu droit ─────────────────────────────────────────────────────────
    with content_col:
        if get_lang() == "en":
            st.markdown(
                """
                ### How to use interactive charts

                1. Choose **a Y variable** (numeric or categorical) and optionally **an X variable**.
                2. The X/Y combination will automatically determine:
                   - The chart type (histogram, boxplot, stacked bars, scatter plot…)
                   - The **most appropriate statistical test** (normality, ANOVA, Kruskal, Chi², Fisher, correlation…)
                3. The test and p-value are displayed **below the chart**.
                """
            )
        else:
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

        # ── Suggestions de questions intéressantes ──
        with st.expander("💡 " + t("Suggestions d'analyses"), expanded=False):
            if get_lang() == "en":
                st.markdown(
                    """
                    **Questions you can answer with this tool:**

                    | Question | Y variable | X variable |
                    |----------|-----------|-----------|
                    | Are scores different by Style? | Note | Style |
                    | Do certain katas win more? | Victoire | Kata |
                    | Is there a link between age and score? | Note | Age |
                    | Is the gender difference significant? | Note | Sexe |
                    | Does ranking predict the scores? | Note | Ranking |
                    | Are some rounds more competitive? | Note | N_Tour |
                    | Which continents dominate? | Victoire | Continent |
                    | Score distribution overview | Note | — |
                    """
                )
            else:
                st.markdown(
                    """
                    **Questions auxquelles cet outil peut répondre :**

                    | Question | Variable Y | Variable X |
                    |----------|-----------|-----------|
                    | Les notes diffèrent-elles par style ? | Note | Style |
                    | Certains katas gagnent-ils plus ? | Victoire | Kata |
                    | Y a-t-il un lien entre âge et notes ? | Note | Age |
                    | Différence H/F significative ? | Note | Sexe |
                    | Le ranking prédit-il les notes ? | Note | Ranking |
                    | Certains tours sont-ils plus compétitifs ? | Note | N_Tour |
                    | Quels continents dominent ? | Victoire | Continent |
                    | Distribution des notes | Note | — |
                    """
                )

        st.subheader(t("Type de graphique généré"))

        # ─── Cas 1 : une seule variable numérique en Y ───────────────────────
        if y_num != _NONE and x_num == _NONE and x_cat == _NONE and y_cat == _NONE:
            st.markdown(f"**{t('Distribution de la variable')} `{fmt_col(y_num)}`**")

            with filters_col:
                filter_panel_open()
                st.markdown(f"#### {t('Filtre (optionnel)')}")
                filtre_var = st.selectbox(
                    t("Variable catégorielle pour filtrer"),
                    [_NONE] + variables_categorielles, key="graphs_filter_num_only",
                )
                if filtre_var != _NONE:
                    modalites = data_compet[filtre_var].dropna().unique().tolist()
                    modalites_selectionnees = st.multiselect(
                        f"{t('Modalités de')} {filtre_var}", modalites, default=modalites, key="graphs_modalites_num_only",
                    )
                filter_panel_close()

            data_filtered = _apply_optional_filter(data_compet, filtre_var, modalites_selectionnees, _NONE)
            data_filtered = data_filtered[data_filtered[y_num].notna()]

            if data_filtered.empty:
                st.warning(t("Aucune donnée disponible avec ces filtres."))
                return

            fig = px.histogram(data_filtered, x=y_num, nbins=30, marginal="box")
            fig.add_vline(x=data_filtered[y_num].mean(), line_dash="dash", line_color="red",
                          annotation_text=t("Moyenne"), annotation_position="top left")
            fig.add_vline(x=data_filtered[y_num].median(), line_dash="dash", line_color="green",
                          annotation_text=t("Médiane"), annotation_position="top right")
            st.plotly_chart(fig, width="stretch")

            st.subheader(t("Test de normalité (automatique)"))
            if st.button(t("Effectuer le test de normalité"), key="graphs_btn_normality"):
                _display_normality(data_filtered[y_num], y_num)

        # ─── Cas 2 : Y numérique, X catégorielle ─────────────────────────────
        elif y_num != _NONE and x_cat != _NONE and x_num == _NONE and y_cat == _NONE:
            st.markdown(f"**{t('Distribution de')} `{fmt_col(y_num)}` {t('par rapport à chaque modalité de')} `{fmt_col(x_cat)}`**")

            with filters_col:
                filter_panel_open()
                st.markdown(f"#### {t('Filtre (optionnel)')}")
                filtre_var = st.selectbox(
                    t("Variable catégorielle pour filtrer"),
                    [_NONE] + variables_categorielles, key="graphs_filter_y_num_x_cat",
                )
                if filtre_var != _NONE:
                    modalites = data_compet[filtre_var].dropna().unique().tolist()
                    modalites_selectionnees = st.multiselect(
                        f"{t('Modalités de')} {filtre_var}", modalites, default=modalites, key="graphs_modalites_y_num_x_cat",
                    )
                filter_panel_close()

            data_filtered = _apply_optional_filter(data_compet, filtre_var, modalites_selectionnees, _NONE)
            data_filtered = data_filtered[data_filtered[y_num].notna()]

            if data_filtered.empty:
                st.warning(t("Aucune donnée disponible avec ces filtres."))
                return

            fig = px.box(data_filtered, x=x_cat, y=y_num, points="all")
            st.plotly_chart(fig, width="stretch")

            st.subheader(t("Test Statistique (automatique)"))
            if st.button(t("Effectuer le test statistique"), key="graphs_btn_y_num_x_cat"):
                _display_oneway(data_filtered, y_num, x_cat)

        # ─── Cas 3 : une seule variable catégorielle en Y ────────────────────
        elif y_cat != _NONE and x_num == _NONE and x_cat == _NONE and y_num == _NONE:
            with filters_col:
                filter_panel_open()
                st.markdown(f"#### {t('Filtre (optionnel)')}")
                filtre_var = st.selectbox(
                    t("Variable catégorielle pour filtrer"),
                    [_NONE] + variables_categorielles, key="graphs_filter_y_cat_only",
                )
                if filtre_var != _NONE:
                    modalites = data_compet[filtre_var].dropna().unique().tolist()
                    modalites_selectionnees = st.multiselect(
                        f"{t('Modalités de')} {filtre_var}", modalites, default=modalites, key="graphs_modalites_y_cat_only",
                    )
                filter_panel_close()

            data_filtered = _apply_optional_filter(data_compet, filtre_var, modalites_selectionnees, _NONE)

            st.markdown(f"**{t('Histogramme des effectifs de chaque modalité de')} `{fmt_col(y_cat)}`**")
            counts = data_filtered[y_cat].value_counts().reset_index()
            counts.columns = [y_cat, t("Effectif")]
            st.plotly_chart(px.bar(counts, x=y_cat, y=t("Effectif")), width="stretch")

            st.markdown(f"**{t('Histogramme des proportions de chaque modalité de')} `{fmt_col(y_cat)}`**")
            counts_prop = data_filtered[y_cat].value_counts(normalize=True).reset_index()
            counts_prop.columns = [y_cat, "Proportion"]
            st.plotly_chart(px.bar(counts_prop, x=y_cat, y="Proportion"), width="stretch")

        # ─── Cas 4 : X et Y catégorielles ────────────────────────────────────
        elif y_cat != _NONE and x_cat != _NONE and y_num == _NONE and x_num == _NONE:
            st.markdown(f"**{t('Proportions des modalités de')} `{fmt_col(y_cat)}` {t('en fonction de')} `{fmt_col(x_cat)}`**")

            with filters_col:
                filter_panel_open()
                st.markdown(f"#### {t('Filtre (optionnel)')}")
                filtre_var = st.selectbox(
                    t("Variable catégorielle pour filtrer"),
                    [_NONE] + variables_categorielles, key="graphs_filter_x_y_cat",
                )
                if filtre_var != _NONE:
                    modalites = data_compet[filtre_var].dropna().unique().tolist()
                    modalites_selectionnees = st.multiselect(
                        f"{t('Modalités de')} {filtre_var}", modalites, default=modalites, key="graphs_modalites_x_y_cat",
                    )
                filter_panel_close()

            data_filtered = _apply_optional_filter(data_compet, filtre_var, modalites_selectionnees, _NONE)
            if data_filtered.empty:
                st.warning(t("Aucune donnée disponible avec ces filtres."))
                return

            crosstab = pd.crosstab(data_filtered[x_cat], data_filtered[y_cat], normalize="index")
            crosstab.reset_index(inplace=True)
            crosstab_melted = crosstab.melt(id_vars=x_cat, var_name=y_cat, value_name="Proportion")
            st.plotly_chart(px.bar(crosstab_melted, x=x_cat, y="Proportion", color=y_cat, barmode="stack"), width="stretch")

            st.subheader(t("Test Statistique (automatique)"))
            if st.button(t("Effectuer le test d'indépendance"), key="graphs_btn_x_y_cat"):
                _display_chi2(data_filtered, x_cat, y_cat)

        # ─── Cas 5 : X et Y numériques ───────────────────────────────────────
        elif x_num != _NONE and y_num != _NONE and x_cat == _NONE and y_cat == _NONE:
            st.markdown(f"**{t('Nuage de points entre')} `{fmt_col(x_num)}` {t('et')} `{fmt_col(y_num)}`**")

            with filters_col:
                filter_panel_open()
                st.markdown(f"#### {t('Filtre (optionnel)')}")
                filtre_var = st.selectbox(
                    t("Variable catégorielle pour filtrer"),
                    [_NONE] + variables_categorielles, key="graphs_filter_x_y_num",
                )
                if filtre_var != _NONE:
                    modalites = data_compet[filtre_var].dropna().unique().tolist()
                    modalites_selectionnees = st.multiselect(
                        f"{t('Modalités de')} {filtre_var}", modalites, default=modalites, key="graphs_modalites_x_y_num",
                    )
                filter_panel_close()

            data_filtered = _apply_optional_filter(data_compet, filtre_var, modalites_selectionnees, _NONE)
            data_filtered = data_filtered[data_filtered[x_num].notna() & data_filtered[y_num].notna()]

            if data_filtered.empty:
                st.warning(t("Aucune donnée disponible avec ces filtres."))
                return

            st.plotly_chart(px.scatter(data_filtered, x=x_num, y=y_num), width="stretch")

            st.subheader(t("Test de corrélation (automatique)"))
            if st.button(t("Effectuer le test de corrélation"), key="graphs_btn_corr"):
                _display_correlation(data_filtered, x_num, y_num)

        else:
            st.warning(t("Veuillez sélectionner des combinaisons cohérentes de variables."))


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers (display wrappers)
# ═══════════════════════════════════════════════════════════════════════════════

def _apply_optional_filter(data_compet, filtre_var, modalites_selectionnees, none_val=None):
    _none = none_val if none_val is not None else t("Aucune")
    if filtre_var != _none and modalites_selectionnees is not None:
        return data_compet[data_compet[filtre_var].isin(modalites_selectionnees)].copy()
    return data_compet.copy()


def _display_normality(series, var_name):
    res = normality_test_auto(series)
    if np.isnan(res["stat"]) or np.isnan(res["p"]):
        st.warning(res["comment"])
    else:
        st.write(
            f"{t('Test utilisé')} : **{res['test_name']}**  \n"
            f"- {t('Statistique')} : **{res['stat']:.3f}**  \n"
            f"- p-value : **{fmt_p(res['p'])}**"
        )
        if res["p"] > 0.05:
            st.write(t("La normalité **n'est pas rejetée** pour") + f" **{var_name}** (p = {fmt_p(res['p'])}).")
        else:
            st.write(t("La normalité est **rejetée** pour") + f" **{var_name}** (p = {fmt_p(res['p'])}).")
        st.markdown(res["comment"])


def _display_oneway(data_filtered, y_num, x_cat):
    res = oneway_test_auto(data_filtered, y_num, x_cat)
    if np.isnan(res["stat"]) or np.isnan(res["p"]):
        st.warning(res["comment"])
    else:
        st.write(
            f"{t('Test utilisé')} : **{res['test_name']}**  \n"
            f"- {t('Statistique')} : **{res['stat']:.3f}**  \n"
            f"- p-value : **{fmt_p(res['p'])}**"
        )
        if res["p"] < 0.05:
            st.write(t("Différence **significative** de") + f" **{y_num}** " + t("entre les groupes de") + f" **{x_cat}** (p = {fmt_p(res['p'])}).")
        else:
            st.write(t("Aucune différence significative de") + f" **{y_num}** " + t("entre les groupes de") + f" **{x_cat}** (p = {fmt_p(res['p'])}).")
        st.markdown(res["comment"])


def _display_chi2(data_filtered, x_cat, y_cat):
    res = chi2_or_fisher_auto(data_filtered, x_cat, y_cat)
    if np.isnan(res["stat"]) or np.isnan(res["p"]):
        st.warning(res["comment"])
    else:
        st.write(f"{t('Test utilisé')} : **{res['test_name']}**  \n- {t('Statistique')} : **{res['stat']:.3f}**")
        if res["p"] is not None:
            st.write(f"- p-value : **{fmt_p(res['p'])}**")
        if res["dof"] is not None:
            st.write(f"- {t('Degrés de liberté')} : **{res['dof']}**")
        if res["p"] is not None:
            if res["p"] < 0.05:
                st.write(t("Association **significative** entre") + f" **{x_cat}** " + t("et") + f" **{y_cat}** (p = {fmt_p(res['p'])}).")
            else:
                st.write(t("Aucune association significative entre") + f" **{x_cat}** " + t("et") + f" **{y_cat}** (p = {fmt_p(res['p'])}).")
        st.markdown(res["comment"])


def _display_correlation(data_filtered, x_num, y_num):
    res = correlation_auto(data_filtered[x_num], data_filtered[y_num])
    if not res["tests"]:
        st.warning(res["comment"])
    else:
        for test in res["tests"]:
            st.write(
                f"Test : **{test['name']}**  \n"
                f"- {t('Coefficient')} : **{test['r']:.3f}**  \n"
                f"- p-value : **{fmt_p(test['p'])}**"
            )
            if test["p"] < 0.05:
                st.write(t("Corrélation **significative**") + f" (p = {fmt_p(test['p'])}).")
            else:
                st.write(t("Pas de corrélation significative") + f" (p = {fmt_p(test['p'])}).")
            st.markdown(test["comment"])
