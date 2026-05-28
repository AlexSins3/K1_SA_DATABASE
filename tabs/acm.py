# tabs/acm.py

from io import StringIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import prince

from constants.tours import classify_tour_type
from utils.ui import filter_panel_open, filter_panel_close
from utils.data_helpers import normalize_victoire_category
from utils.interpretations import show_tab_help, _color_badge
from utils.lang import t, get_lang


def _split_var_mod(index_name: str, variables: list[str]):
    """Decompose ``'Kata_Unsu'`` into ``('Kata', 'Unsu')`` using the MCA variable list."""
    for var in variables:
        prefix = f"{var}_"
        if index_name.startswith(prefix):
            return var, index_name[len(prefix):]
    parts = index_name.split("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return index_name, ""


@st.fragment
def show_acm_tab(data: pd.DataFrame) -> None:
    st.header(t("Carte des associations Kata / Résultat (ACM)"))
    show_tab_help("acm")

    df = data.copy()

    filters_col, content_col = st.columns([0.9, 2.4])

    data_acm_filtered = df.copy()

    # ══════════════════════════════════════════════════════════════════════════
    # Left column: filters
    # ══════════════════════════════════════════════════════════════════════════
    with filters_col:
        filter_panel_open()
        st.markdown(t("### 🎯 Filtres ACM"))

        # ── Type de compétition ──
        type_compet_options = [t("Tous"), "Premier League (K1)", "Series A (SA)"]
        selected_type_compet = st.radio(
            t("Type de compétition"), type_compet_options, key="acm_type_compet",
        )
        if selected_type_compet == "Premier League (K1)":
            data_acm_filtered = data_acm_filtered[data_acm_filtered["Type_Compet"] == "K1"]
        elif selected_type_compet == "Series A (SA)":
            data_acm_filtered = data_acm_filtered[data_acm_filtered["Type_Compet"] == "SA"]

        # ── Année(s) ──
        if "Year" in data_acm_filtered.columns:
            years_acm = sorted(pd.to_numeric(data_acm_filtered["Year"], errors="coerce").dropna().unique().tolist())
            if years_acm:
                sel_years_acm = st.multiselect(
                    t("Année(s)"), [int(y) for y in years_acm],
                    default=[int(y) for y in years_acm], key="acm_years"
                )
                if sel_years_acm:
                    data_acm_filtered = data_acm_filtered[
                        pd.to_numeric(data_acm_filtered["Year"], errors="coerce").isin([float(y) for y in sel_years_acm])
                    ]

        # ── Type de Tour ──
        data_acm_filtered["Type de Tour"] = data_acm_filtered["N_Tour"].apply(classify_tour_type)
        data_acm_filtered = data_acm_filtered.dropna(subset=["Type de Tour"])

        type_de_tour_modalities = sorted(data_acm_filtered["Type de Tour"].unique().tolist())
        selected_types = st.multiselect(
            t("Type(s) de Tour à inclure"),
            options=type_de_tour_modalities,
            default=type_de_tour_modalities,
            key="acm_type_tour",
        )
        if not selected_types:
            st.warning(t("Aucun 'Type de Tour' sélectionné. Veuillez en sélectionner au moins un."))
            filter_panel_close()
            return

        data_acm_filtered = data_acm_filtered[data_acm_filtered["Type de Tour"].isin(selected_types)]

        # ── Sexe ──
        st.markdown("---")
        st.markdown(t("#### 🧑 Sexe"))
        sexe_modalities = [t("Aucun"), "M", "F"]
        selected_sexe = st.selectbox(t("Sexe à inclure"), sexe_modalities, index=0, key="acm_sexe")
        if selected_sexe != t("Aucun"):
            data_acm_filtered = data_acm_filtered[data_acm_filtered["Sexe"] == selected_sexe]
            if data_acm_filtered.empty:
                st.warning(t("Aucune donnée disponible pour le sexe sélectionné."))
                filter_panel_close()
                return

        # ── Style ──
        st.markdown("---")
        st.markdown(t("#### 🧬 Style de kata"))

        style_selected = None
        if "Style" not in data_acm_filtered.columns:
            st.info(t("Pas de colonne 'Style' dans les données : filtrage par style désactivé."))
        else:
            tmp_style = data_acm_filtered[["Kata", "Style"]].dropna()
            if tmp_style.empty:
                st.info(t("Aucune information de style disponible après filtrage."))
            else:
                style_values = sorted(tmp_style["Style"].dropna().unique().tolist())
                style_selected = st.multiselect(
                    t("Style(s) à inclure"), options=style_values, default=style_values, key="acm_style",
                )

                style_per_kata = tmp_style.groupby("Kata")["Style"].apply(lambda s: sorted(set(s.dropna())))
                ambiguous_katas = style_per_kata[style_per_kata.apply(len) > 1]
                ambiguous_except_suparinpei = ambiguous_katas[ambiguous_katas.index != "Suparinpei"]

                if not ambiguous_except_suparinpei.empty:
                    if get_lang() == "en":
                        msg = (
                            "Error: some katas are associated with multiple styles "
                            "(not allowed, except for **Suparinpei**): "
                        )
                    else:
                        msg = (
                            "Erreur : certains katas sont associés à plusieurs styles "
                            "(ce n'est pas autorisé, sauf pour **Suparinpei**) : "
                        )
                    msg += ", ".join(
                        f"{kata} ({'/'.join(styles)})" for kata, styles in ambiguous_except_suparinpei.items()
                    )
                    st.error(msg)
                    filter_panel_close()
                    return

                if style_selected:
                    allowed_katas = []
                    for kata, styles in style_per_kata.items():
                        if kata == "Suparinpei":
                            allowed_katas.append(kata)
                        elif len(styles) == 1 and styles[0] in style_selected:
                            allowed_katas.append(kata)
                    data_acm_filtered = data_acm_filtered[data_acm_filtered["Kata"].isin(allowed_katas)]
                else:
                    st.warning(t("Aucun style sélectionné, tous les styles sont inclus par défaut."))

        if data_acm_filtered.empty:
            st.warning(t("Aucune donnée disponible après filtrage par style."))
            filter_panel_close()
            return

        # ── Katas ──
        st.markdown("---")
        st.markdown(t("#### 🥋 Katas"))

        kata_modalities = sorted(data_acm_filtered["Kata"].dropna().unique().tolist())
        if not kata_modalities:
            st.warning(t("Aucun kata disponible après filtrage."))
            filter_panel_close()
            return

        selected_katas = st.multiselect(
            t("Katas à inclure dans l'ACM"), options=kata_modalities, default=kata_modalities, key="acm_katas",
        )
        if not selected_katas:
            st.warning(t("Aucun kata sélectionné. Veuillez en sélectionner au moins un."))
            filter_panel_close()
            return

        data_acm_filtered = data_acm_filtered[data_acm_filtered["Kata"].isin(selected_katas)]
        if data_acm_filtered.empty:
            st.warning(t("Aucune donnée disponible après filtrage par katas."))
            filter_panel_close()
            return

        # ── Normalisation Victoire ──
        data_acm_filtered["Victoire_norm"] = data_acm_filtered["Victoire"].apply(normalize_victoire_category)

        # ── Options d'affichage ──
        st.markdown("---")
        st.markdown(t("#### 🎨 Options d'affichage"))

        display_individuals = st.checkbox(t("Afficher les individus"), value=False, key="acm_display_individuals")
        display_modalities = st.checkbox(t("Afficher les modalités des variables"), value=True, key="acm_display_modalities")

        filter_panel_close()

    # ══════════════════════════════════════════════════════════════════════════
    # Right column: ACM computation & display
    # ══════════════════════════════════════════════════════════════════════════
    with content_col:
        st.subheader(t("Paramètres de l'ACM"))
        if get_lang() == "en":
            st.markdown(
                """
                The MCA below is performed on qualitative variables:
                - **Kata**
                - **Round** (competition round)
                - **Victory** (True / False, normalized)

                The filters on the left allow you to **restrict the analysis scope**
                (competition type, round type, gender, style, katas).
                """
            )
        else:
            st.markdown(
                """
                L'ACM ci-dessous est réalisée sur les variables qualitatives :
                - **Kata**
                - **Tour** (tour de la compétition)
                - **Victoire** (True / False, normalisée)

                Les filtres à gauche permettent de **restreindre le périmètre d'analyse**
                (type de compétition, type de tour, sexe, style, katas).
                """
            )

        mca_variables = ["Kata", "N_Tour", "Victoire_norm"]
        var_labels = {"Kata": "Kata", "N_Tour": "Tour", "Victoire_norm": "Victoire"}

        data_mca = data_acm_filtered[mca_variables].copy().dropna()
        if data_mca.empty:
            st.warning(t("Aucune donnée disponible après suppression des valeurs manquantes. Impossible de réaliser l'ACM."))
            return

        for col in mca_variables:
            data_mca[col] = data_mca[col].astype(str).astype("category")

        if "Nom" in data_acm_filtered.columns:
            names_for_rows = data_acm_filtered.loc[data_mca.index, "Nom"].astype(object)
            names_for_rows = names_for_rows.where(names_for_rows.notna(), t("Inconnu")).astype(str)
        else:
            names_for_rows = data_mca.index.astype(str)

        # ── Fit MCA ──
        mca = prince.MCA(n_components=2, random_state=42)
        mca = mca.fit(data_mca)

        modalities_coords = mca.column_coordinates(data_mca)

        variables_list = []
        modalites_list = []
        for idx in modalities_coords.index:
            var_name, mod = _split_var_mod(idx, mca_variables)
            variables_list.append(var_name)
            modalites_list.append(mod)
        modalities_coords["Variable"] = variables_list
        modalities_coords["Modalité"] = modalites_list

        if display_individuals:
            individuals_coords = mca.row_coordinates(data_mca)

        eigenvalues = mca.eigenvalues_
        total_inertia = eigenvalues.sum()
        explained_inertia = eigenvalues / total_inertia

        # ── Plot ──
        st.subheader(t("Représentation de l'ACM"))
        fig = go.Figure()

        if display_modalities:
            for var in mca_variables:
                label = var_labels[var]
                vc = modalities_coords[modalities_coords["Variable"] == var]
                _trace_name = f"Modalities of {label}" if get_lang() == "en" else f"Modalités de {label}"
                fig.add_trace(go.Scatter(
                    x=vc[0], y=vc[1], mode="markers+text",
                    name=_trace_name, text=vc["Modalité"],
                    textposition="top center", marker=dict(size=10),
                ))

        if display_individuals:
            fig.add_trace(go.Scatter(
                x=individuals_coords[0], y=individuals_coords[1], mode="markers",
                name=t("Individus"), marker=dict(size=5, color="grey", opacity=0.5),
                text=names_for_rows, hoverinfo="text",
            ))

        if display_individuals:
            x_coords = pd.concat([modalities_coords[0], individuals_coords[0]])
            y_coords = pd.concat([modalities_coords[1], individuals_coords[1]])
        else:
            x_coords = modalities_coords[0]
            y_coords = modalities_coords[1]

        x_min, x_max = x_coords.min() - 0.5, x_coords.max() + 0.5
        y_min, y_max = y_coords.min() - 0.5, y_coords.max() + 0.5

        fig.update_layout(
            title=t("Carte factorielle de l'ACM"),
            xaxis_title=f"Dimension 1 ({explained_inertia[0] * 100:.2f}%)",
            yaxis_title=f"Dimension 2 ({explained_inertia[1] * 100:.2f}%)",
            showlegend=True, width=900, height=650,
        )
        fig.update_xaxes(range=[x_min, x_max], zeroline=False)
        fig.update_yaxes(range=[y_min, y_max], zeroline=False)

        fig.add_shape(type="line", x0=x_min, y0=0, x1=x_max, y1=0, line=dict(color="black", width=1))
        fig.add_shape(type="line", x0=0, y0=y_min, x1=0, y1=y_max, line=dict(color="black", width=1))

        st.plotly_chart(fig, width="stretch")

        # ── Qualité de la carte ──
        total_explained_pct = (explained_inertia[0] + explained_inertia[1]) * 100
        if total_explained_pct >= 50:
            quality_badge = _color_badge(f"{total_explained_pct:.1f}% — {t('Bonne qualité')}", "green")
        elif total_explained_pct >= 30:
            quality_badge = _color_badge(f"{total_explained_pct:.1f}% — {t('Qualité correcte')}", "orange")
        else:
            quality_badge = _color_badge(f"{total_explained_pct:.1f}% — {t('Qualité faible')}", "red")
        st.markdown(f"**{t('Inertie captée par les 2 axes')}** : {quality_badge}", unsafe_allow_html=True)

        # ── Key findings (always visible) ──
        st.subheader("🔑 " + t("Résultats clés"))
        _render_key_findings(modalities_coords, mca_variables)

        # ── Detailed interpretation (expandable) ──
        st.subheader(t("Interprétation de l'ACM"))

        with st.expander(t("📖 Interprétation détaillée"), expanded=False):
            interpretation = _interpret_acm(mca, data_mca, mca_variables)
            st.markdown(interpretation)

        with st.expander(t("💡 Comment lire cette carte ?"), expanded=False):
            if get_lang() == "en":
                st.markdown(
                    """
                    **Multiple Correspondence Analysis (MCA)** explores relationships between multiple
                    **qualitative** variables (here: *Kata*, *Round*, *Victory*).

                    ### How to read the chart

                    - Each **modality** (e.g. a kata name, a round, `True/False` for victory) is represented by a point.
                    - Modalities that are **close** on the factor map tend to appear **together** in the same matches.
                    - **Dimension 1** (horizontal axis) and **Dimension 2** (vertical axis) are the two directions that best summarize the data variability.
                    - The modalities **`Victory = True`** and **`Victory = False`** serve as reference points:
                        - Katas close to **`Victory = True`** are mostly associated with won matches.
                        - Katas close to **`Victory = False`** are mostly associated with lost matches.

                    ### Keep in mind

                    - MCA shows **associations**, not direct causality.
                    - Results strongly depend on the **filters** chosen on the left (competition type, round types, gender, style, katas).
                    - To refine the analysis, you can:
                        - restrict to one gender,
                        - keep only certain round types,
                        - filter by style (Shotokan / Shitoryu),
                        - exclude very infrequent katas, etc.

                    You can then click **"Automatically interpret MCA"** for a guided reading of katas
                    most associated with victories and defeats.
                    """
                )
            else:
                st.markdown(
                    """
                    L'**analyse des correspondances multiples (ACM)** sert à explorer les relations entre plusieurs variables
                    **qualitatives** (ici : *Kata*, *Tour*, *Victoire*).

                    ### Comment lire le graphique

                    - Chaque **modalité** (par ex. un nom de kata, un tour, `True/False` pour la victoire) est représentée par un point.
                    - Les modalités **proches** sur le plan factoriel ont tendance à apparaître **ensemble** dans les mêmes combats.
                    - La **Dimension 1** (axe horizontal) et la **Dimension 2** (axe vertical) sont les deux directions qui résument le mieux
                      la variabilité des données.
                    - Les modalités **`Victoire = True`** et **`Victoire = False`** servent de repères :
                        - Les katas proches de **`Victoire = True`** sont plutôt associés à des combats gagnés.
                        - Les katas proches de **`Victoire = False`** sont plutôt associés à des combats perdus.

                    ### À garder en tête

                    - L'ACM montre des **associations**, pas une causalité directe.
                    - Les résultats dépendent fortement des **filtres** choisis à gauche (type de compétition, types de tours, sexe, style, katas).
                    - Pour affiner l'analyse, tu peux :
                        - restreindre à un seul sexe,
                        - ne garder que certains types de tours,
                        - filtrer par style (Shotokan / Shitoryu),
                        - exclure des katas très peu fréquents, etc.

                    Tu peux ensuite cliquer sur **"Interpréter automatiquement l'ACM"** pour obtenir une lecture guidée des katas
                    les plus associés aux victoires et aux défaites.
                    """
                )


# ═══════════════════════════════════════════════════════════════════════════════
# Key findings — always visible summary
# ═══════════════════════════════════════════════════════════════════════════════

def _render_key_findings(modalities_coords, mca_vars):
    """Display a quick summary: katas closest to Victory=True and Victory=False."""
    kata_coords = modalities_coords[modalities_coords["Variable"] == "Kata"].copy()
    victoire_coords = modalities_coords[modalities_coords["Variable"] == "Victoire_norm"].copy()

    if kata_coords.empty or victoire_coords.empty:
        st.info(t("Pas assez de données pour extraire les résultats clés."))
        return

    # Find True/False modalities
    mods_lower = victoire_coords["Modalité"].astype(str).str.strip().str.lower()
    vic_true = victoire_coords[mods_lower.isin(["true", "1", "vrai", "oui"])]
    vic_false = victoire_coords[mods_lower.isin(["false", "0", "faux", "non"])]

    if vic_true.empty or vic_false.empty:
        st.info(t("Modalités Victoire True/False non identifiables."))
        return

    vt = vic_true.iloc[0]
    vf = vic_false.iloc[0]

    # Distance de chaque kata à True et False
    kata_coords["Dist_True"] = np.sqrt((kata_coords[0] - vt[0]) ** 2 + (kata_coords[1] - vt[1]) ** 2)
    kata_coords["Dist_False"] = np.sqrt((kata_coords[0] - vf[0]) ** 2 + (kata_coords[1] - vf[1]) ** 2)
    kata_coords["Score"] = kata_coords["Dist_False"] - kata_coords["Dist_True"]
    kata_coords = kata_coords.sort_values("Score", ascending=False)

    col_win, col_lose = st.columns(2)
    with col_win:
        st.markdown(f"**🏆 {t('Katas associés à la victoire')}**")
        for _, row in kata_coords.head(5).iterrows():
            st.markdown(f"- {row['Modalité']}")
    with col_lose:
        st.markdown(f"**❌ {t('Katas associés à la défaite')}**")
        for _, row in kata_coords.tail(5).iterrows():
            st.markdown(f"- {row['Modalité']}")

    # Tours associés
    tour_coords = modalities_coords[modalities_coords["Variable"] == "N_Tour"].copy()
    if not tour_coords.empty:
        tour_coords["Dist_True"] = np.sqrt((tour_coords[0] - vt[0]) ** 2 + (tour_coords[1] - vt[1]) ** 2)
        tour_coords["Dist_False"] = np.sqrt((tour_coords[0] - vf[0]) ** 2 + (tour_coords[1] - vf[1]) ** 2)
        tour_coords["Score"] = tour_coords["Dist_False"] - tour_coords["Dist_True"]
        tour_coords = tour_coords.sort_values("Score", ascending=False)

        st.markdown(f"**🏟️ {t('Tours les plus associés à la victoire')}** : "
                    f"{', '.join(tour_coords.head(3)['Modalité'].tolist())}")


# ═══════════════════════════════════════════════════════════════════════════════
# Automatic interpretation (extracted from the original inline function)
# ═══════════════════════════════════════════════════════════════════════════════

_TRUE_LIKE = {"true", "1", "vrai", "oui", "y", "o", "yes"}
_FALSE_LIKE = {"false", "0", "faux", "non", "n", "no"}


def _interpret_acm(mca_obj, data_mca_local, mca_vars):
    output = StringIO()
    _en = get_lang() == "en"

    # 1. Variance explained
    eigen = mca_obj.eigenvalues_
    total_inertie = eigen.sum()
    explained = eigen / total_inertie

    if _en:
        output.write("### 1. Variance explained by dimensions\n\n")
        for i, eig in enumerate(explained):
            output.write(f"- Dimension {i + 1}: **{eig * 100:.2f}%** of explained variance.\n")
        output.write("\n---\n")
        output.write("### 2. Kata association with victory / defeat\n\n")
    else:
        output.write("### 1. Variance expliquée par les dimensions\n\n")
        for i, eig in enumerate(explained):
            output.write(f"- Dimension {i + 1} : **{eig * 100:.2f}%** de la variance expliquée.\n")
        output.write("\n---\n")
        output.write("### 2. Association des katas avec la victoire / défaite\n\n")

    coords = mca_obj.column_coordinates(data_mca_local)
    coords.columns = [f"Dim_{i + 1}" for i in range(coords.shape[1])]

    variables = []
    modalites = []
    index_raw = []
    for idx in coords.index:
        var_name, mod = _split_var_mod(idx, mca_vars)
        variables.append(var_name)
        modalites.append(mod)
        index_raw.append(str(idx))

    coords["Variable"] = variables
    coords["Modalité"] = modalites
    coords["Index_raw"] = index_raw

    kata_coords = coords[coords["Variable"] == "Kata"].copy()
    victoire_coords = coords[coords["Variable"] == "Victoire_norm"].copy()

    if victoire_coords.empty:
        if _en:
            output.write(
                "Unable to identify the modalities of the `Victory` variable "
                "in this configuration (after filtering and normalization).\n\n"
            )
        else:
            output.write(
                "Impossible d'identifier les modalités de la variable `Victoire` "
                "dans cette configuration (après filtrage et normalisation).\n\n"
            )
    else:
        mods_raw = victoire_coords["Modalité"].astype(str).str.strip().str.lower()
        idx_raw_lower = victoire_coords["Index_raw"].astype(str).str.strip().str.lower()

        is_true = mods_raw.isin(_TRUE_LIKE) | idx_raw_lower.str.contains("true")
        is_false = mods_raw.isin(_FALSE_LIKE) | idx_raw_lower.str.contains("false")

        victoire_coords = victoire_coords.copy()
        victoire_coords["is_true"] = is_true
        victoire_coords["is_false"] = is_false

        vic_true = victoire_coords[victoire_coords["is_true"]]
        vic_false = victoire_coords[victoire_coords["is_false"]]

        if not vic_true.empty and not vic_false.empty:
            _write_full_proximity(output, kata_coords, vic_true.iloc[0], vic_false.iloc[0], _en)
        elif not vic_true.empty:
            if _en:
                output.write(
                    "With current filters, observed matches are almost all on the "
                    "**`Victory = True`** side (no clear `False` modality).\n\n"
                    "We can still list the katas closest to this victory modality:\n\n"
                )
            else:
                output.write(
                    "Avec les filtres actuels, les combats observés sont quasiment "
                    "tous du côté **`Victoire = True`** (aucune modalité claire `False`).\n\n"
                    "On peut tout de même lister les katas les plus proches de cette modalité de victoire :\n\n"
                )
            _write_distance_to_single(output, kata_coords, vic_true.iloc[0], "Victoire = True", _en)
        elif not vic_false.empty:
            if _en:
                output.write(
                    "With current filters, observed matches are almost all on the "
                    "**`Victory = False`** side (no clear `True` modality).\n\n"
                    "We can still list the katas closest to this defeat modality:\n\n"
                )
            else:
                output.write(
                    "Avec les filtres actuels, les combats observés sont quasiment "
                    "tous du côté **`Victoire = False`** (aucune modalité claire `True`).\n\n"
                    "On peut tout de même lister les katas les plus proches de cette modalité de défaite :\n\n"
                )
            _write_distance_to_single(output, kata_coords, vic_false.iloc[0], "Victoire = False", _en)
        else:
            if _en:
                output.write(
                    "The modalities of the `Victory` variable cannot be clearly "
                    "linked to `True` or `False`.\n\n"
                    "We can however identify the most **extreme katas on Dimension 1**:\n\n"
                )
            else:
                output.write(
                    "Les modalités de la variable `Victoire` ne peuvent pas être "
                    "reliées clairement à `True` ou `False`.\n\n"
                    "On peut toutefois repérer les katas les plus **extrêmes sur la Dimension 1** :\n\n"
                )
            if not kata_coords.empty:
                kata_ext = kata_coords.copy()
                kata_ext["Abs_Dim1"] = kata_ext["Dim_1"].abs()
                kata_ext = kata_ext.sort_values("Abs_Dim1", ascending=False)
                for _, row in kata_ext.head(5).iterrows():
                    if _en:
                        sens = "positive side" if row["Dim_1"] >= 0 else "negative side"
                        output.write(f"- **{row['Modalité']}** (strongly positioned on Dimension 1, {sens}).\n")
                    else:
                        sens = "côté positif" if row["Dim_1"] >= 0 else "côté négatif"
                        output.write(f"- **{row['Modalité']}** (fortement positionné sur la Dimension 1, {sens}).\n")

    output.write("\n---\n")
    if _en:
        output.write("### 3. How to read these results (simplified)\n\n")
        output.write(
            "- Each **point** on the chart represents either a **modality** "
            "(e.g. a kata, a round, True/False for victory), or a **match** "
            "(if individuals are displayed).\n"
            "- When two modalities are **close**, it means they appear "
            "**often together** in the same matches.\n"
            "- For example, if a kata is positioned near the modality "
            "**`Victory = True`**, it suggests this kata is more often associated "
            "with **won** matches.\n"
            "- Conversely, if it is near **`Victory = False`**, it seems "
            "more associated with **defeats**.\n\n"
            "> ⚠️ Note: MCA highlights **statistical associations**, "
            "not cause and effect. A kata close to `Victory = True` does not "
            "\"win\" on its own, it is simply **frequent** in won matches.\n"
        )
    else:
        output.write("### 3. Comment lire ces résultats (version vulgarisée)\n\n")
        output.write(
            "- Chaque **point** sur le graphique représente soit une **modalité** "
            "(par ex. un kata, un tour, True/False pour la victoire), soit un **combat** "
            "(si les individus sont affichés).\n"
            "- Quand deux modalités sont **proches**, cela signifie qu'elles apparaissent "
            "**souvent ensemble** dans les mêmes combats.\n"
            "- Par exemple, si un kata est positionné près de la modalité "
            "**`Victoire = True`**, cela suggère que ce kata est plus souvent associé "
            "à des combats **gagnés**.\n"
            "- À l'inverse, s'il est situé près de **`Victoire = False`**, il semble "
            "davantage associé à des **défaites**.\n\n"
            "> ⚠️ Attention : l'ACM met en évidence des **associations statistiques**, "
            "pas une relation de cause à effet. Un kata proche de `Victoire = True` ne "
            "fait pas \"gagner\" à lui seul, il est simplement **fréquent** dans les combats gagnés.\n"
        )

    return output.getvalue()


def _write_full_proximity(output, kata_coords, victoire_true, victoire_false, _en=False):
    kata_coords = kata_coords.copy()
    kata_coords["Distance_to_Victoire_True"] = np.sqrt(
        (kata_coords["Dim_1"] - victoire_true["Dim_1"]) ** 2
        + (kata_coords["Dim_2"] - victoire_true["Dim_2"]) ** 2
    )
    kata_coords["Distance_to_Victoire_False"] = np.sqrt(
        (kata_coords["Dim_1"] - victoire_false["Dim_1"]) ** 2
        + (kata_coords["Dim_2"] - victoire_false["Dim_2"]) ** 2
    )
    kata_coords["Proximity"] = kata_coords["Distance_to_Victoire_False"] - kata_coords["Distance_to_Victoire_True"]
    kata_coords = kata_coords.sort_values("Proximity", ascending=True)

    if _en:
        output.write("Katas closest to **Victory = True** (potentially associated with victories):\n\n")
    else:
        output.write("Les katas les plus proches de **Victoire = True** (potentiellement associés aux victoires) :\n\n")
    for _, row in kata_coords.head(5).iterrows():
        output.write(f"- **{row['Modalité']}** (proximity = {row['Proximity']:.2f})\n")

    if _en:
        output.write("\nKatas closest to **Victory = False** (potentially associated with defeats):\n\n")
    else:
        output.write("\nLes katas les plus proches de **Victoire = False** (potentiellement associés aux défaites) :\n\n")
    for _, row in kata_coords.tail(5).iterrows():
        output.write(f"- **{row['Modalité']}** (proximity = {row['Proximity']:.2f})\n")


def _write_distance_to_single(output, kata_coords, ref_point, label, _en=False):
    kata_coords = kata_coords.copy()
    kata_coords["Distance"] = np.sqrt(
        (kata_coords["Dim_1"] - ref_point["Dim_1"]) ** 2
        + (kata_coords["Dim_2"] - ref_point["Dim_2"]) ** 2
    )
    kata_coords = kata_coords.sort_values("Distance", ascending=True)
    for _, row in kata_coords.head(5).iterrows():
        if _en:
            output.write(f"- **{row['Modalité']}** (close to {label})\n")
        else:
            output.write(f"- **{row['Modalité']}** (proche de {label})\n")
