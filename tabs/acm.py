# tabs/acm.py

from io import StringIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import prince


def split_var_mod(index_name: str, variables: list[str]):
    """
    Décompose un nom du type 'Kata_Unsu' ou 'Victoire_norm_True'
    en (nom_variable, modalité) en s'appuyant sur la liste des variables de l'ACM.
    """
    for var in variables:
        prefix = f"{var}_"
        if index_name.startswith(prefix):
            return var, index_name[len(prefix):]
    # fallback au cas où (ne devrait presque jamais servir)
    parts = index_name.split("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return index_name, ""


def show_acm_tab(data: pd.DataFrame) -> None:
    st.header("Analyse des Correspondances Multiples (ACM)")

    df = data.copy()

    # =========================
    # Mise en page : bande de filtres à gauche
    # =========================
    filters_col, content_col = st.columns([0.9, 2.4])

    # On prépare les variables qui serviront à filtrer les données pour l'ACM
    data_acm_filtered = df.copy()

    # =========================
    # Colonne de gauche : filtres & options d'affichage
    # =========================
    with filters_col:
        # CSS pour styliser la bande de gauche
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
        st.markdown("### 🎯 Filtres ACM")

        # ---- Type de compétition
        type_compet_options = ["Tous", "Premier League (K1)", "Series A (SA)"]
        selected_type_compet = st.radio(
            "Type de compétition",
            type_compet_options,
            key="acm_type_compet",
        )

        if selected_type_compet != "Tous":
            if selected_type_compet == "Premier League (K1)":
                data_acm_filtered = data_acm_filtered[data_acm_filtered["Type_Compet"] == "K1"]
            elif selected_type_compet == "Series A (SA)":
                data_acm_filtered = data_acm_filtered[data_acm_filtered["Type_Compet"] == "SA"]

        # =========================
        # Création / filtre "Type de Tour"
        # =========================
        def create_type_de_tour(row: pd.Series):
            n_tour = row.get("N_Tour")
            if pd.isna(n_tour):
                return None
            if n_tour in ["Bronze", "Finale", "Final"]:
                return "Match de médaille"
            elif n_tour in ["R1", "R2", "PW1", "PW2", "PW3"]:
                return "Quart/Demi/Finale de poule"
            elif n_tour in ["Pool_1", "T1"]:
                return "Tour 1"
            elif n_tour in ["Pool_2", "T2"]:
                return "Tour 2"
            elif n_tour in ["Pool_3", "T3"]:
                return "Tour 3"
            else:
                return None

        data_acm_filtered["Type de Tour"] = data_acm_filtered.apply(create_type_de_tour, axis=1)
        data_acm_filtered = data_acm_filtered.dropna(subset=["Type de Tour"])

        type_de_tour_modalities = sorted(data_acm_filtered["Type de Tour"].unique().tolist())
        selected_types = st.multiselect(
            "Type(s) de Tour à inclure",
            options=type_de_tour_modalities,
            default=type_de_tour_modalities,
            key="acm_type_tour",
        )

        if not selected_types:
            st.warning("Aucun 'Type de Tour' sélectionné. Veuillez en sélectionner au moins un.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        data_acm_filtered = data_acm_filtered[data_acm_filtered["Type de Tour"].isin(selected_types)]

        # =========================
        # Filtre Sexe
        # =========================
        st.markdown("---")
        st.markdown("#### 🧍 Sexe")

        sexe_modalities = ["Aucun", "M", "F"]
        selected_sexe = st.selectbox(
            "Sexe à inclure",
            sexe_modalities,
            index=0,
            key="acm_sexe",
        )

        if selected_sexe != "Aucun":
            data_acm_filtered = data_acm_filtered[data_acm_filtered["Sexe"] == selected_sexe]
            if data_acm_filtered.empty:
                st.warning("Aucune donnée disponible pour le sexe sélectionné.")
                st.markdown("</div>", unsafe_allow_html=True)
                return

        # =========================
        # Filtre Style (Shotokan / ShitoRyu)
        # =========================
        st.markdown("---")
        st.markdown("#### 🧬 Style de kata")

        # On part du principe qu'il existe une colonne 'Style' et que le style est défini par kata
        if "Style" not in data_acm_filtered.columns:
            st.info("Pas de colonne 'Style' dans les données : filtrage par style désactivé.")
            style_selected = None
        else:
            tmp_style = data_acm_filtered[["Kata", "Style"]].dropna()
            if tmp_style.empty:
                st.info("Aucune information de style disponible après filtrage.")
                style_selected = None
            else:
                # Tous les styles possibles
                style_values = sorted(tmp_style["Style"].dropna().unique().tolist())
                style_selected = st.multiselect(
                    "Style(s) à inclure",
                    options=style_values,
                    default=style_values,
                    key="acm_style",
                )

                # Style par kata (liste de styles par kata)
                style_per_kata = (
                    tmp_style.groupby("Kata")["Style"]
                    .apply(lambda s: sorted(set(s.dropna())))
                )

                # Katas ambigus = plusieurs styles
                ambiguous_katas = style_per_kata[style_per_kata.apply(len) > 1]

                # On exclut explicitement Suparinpei de l'erreur (il peut être dans 2 styles)
                ambiguous_except_suparinpei = ambiguous_katas[ambiguous_katas.index != "Suparinpei"]

                if not ambiguous_except_suparinpei.empty:
                    msg = (
                        "Erreur : certains katas sont associés à plusieurs styles "
                        "(ce n'est pas autorisé, sauf pour **Suparinpei**) : "
                    )
                    msg += ", ".join(
                        f"{kata} ({'/'.join(styles)})"
                        for kata, styles in ambiguous_except_suparinpei.items()
                    )
                    st.error(msg)
                    st.markdown("</div>", unsafe_allow_html=True)
                    return

                # Application du filtre style :
                if style_selected:
                    # On garde :
                    # - les katas avec un seul style, et ce style est sélectionné
                    # - Suparinpei (toujours conservé s'il est présent, même multi-style)
                    allowed_katas_style = []
                    for kata, styles in style_per_kata.items():
                        if kata == "Suparinpei":
                            allowed_katas_style.append(kata)
                        elif len(styles) == 1 and styles[0] in style_selected:
                            allowed_katas_style.append(kata)

                    data_acm_filtered = data_acm_filtered[data_acm_filtered["Kata"].isin(allowed_katas_style)]
                else:
                    st.warning("Aucun style sélectionné, tous les styles sont inclus par défaut.")

        if data_acm_filtered.empty:
            st.warning("Aucune donnée disponible après filtrage par style.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        # =========================
        # Filtre sur les Katas
        # =========================
        st.markdown("---")
        st.markdown("#### 🥋 Katas")

        kata_modalities = sorted(data_acm_filtered["Kata"].dropna().unique().tolist())
        if not kata_modalities:
            st.warning("Aucun kata disponible après filtrage.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        selected_katas = st.multiselect(
            "Katas à inclure dans l'ACM",
            options=kata_modalities,
            default=kata_modalities,
            key="acm_katas",
        )

        if not selected_katas:
            st.warning("Aucun kata sélectionné. Veuillez en sélectionner au moins un.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        data_acm_filtered = data_acm_filtered[data_acm_filtered["Kata"].isin(selected_katas)]
        if data_acm_filtered.empty:
            st.warning("Aucune donnée disponible après filtrage par katas.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        # =========================
        # Normalisation de la variable Victoire
        # =========================
        def normalize_victoire(val):
            if pd.isna(val):
                return np.nan
            s = str(val).strip().lower()
            if s in ["true", "vrai", "1", "oui", "o", "y", "yes"]:
                return "True"
            if s in ["false", "faux", "0", "non", "n", "no"]:
                return "False"
            return np.nan

        data_acm_filtered["Victoire_norm"] = data_acm_filtered["Victoire"].apply(normalize_victoire)

        # =========================
        # Options d'affichage de l'ACM
        # =========================
        st.markdown("---")
        st.markdown("#### 🎨 Options d'affichage")

        display_individuals = st.checkbox(
            "Afficher les individus",
            value=False,
            key="acm_display_individuals",
        )
        display_modalities = st.checkbox(
            "Afficher les modalités des variables",
            value=True,
            key="acm_display_modalities",
        )

        st.markdown("</div>", unsafe_allow_html=True)

    # =========================
    # Colonne de droite : ACM (graph + interprétation)
    # =========================
    with content_col:
        st.subheader("Paramètres de l'ACM")

        st.markdown(
            """
            L'ACM ci-dessous est réalisée sur les variables qualitatives :
            - **Kata**
            - **N_Tour** (tour de la compétition)
            - **Victoire** (True / False, normalisée)

            Les filtres à gauche permettent de **restreindre le périmètre d'analyse**  
            (type de compétition, type de tour, sexe, style, katas).
            """
        )

        # =========================
        # Préparation des données pour l'ACM
        # =========================
        mca_variables = ["Kata", "N_Tour", "Victoire_norm"]
        var_labels = {
            "Kata": "Kata",
            "N_Tour": "N_Tour",
            "Victoire_norm": "Victoire",
        }

        data_mca = data_acm_filtered[mca_variables].copy()
        data_mca = data_mca.dropna()

        if data_mca.empty:
            st.warning("Les données sont vides après suppression des valeurs manquantes. Impossible de réaliser l'ACM.")
            return

        for col in mca_variables:
            data_mca[col] = data_mca[col].astype(str).astype("category")

        if "Nom" in data_acm_filtered.columns:
            # on récupère les noms alignés sur les index de data_mca
            names_for_rows = data_acm_filtered.loc[data_mca.index, "Nom"]

            # on passe en objet, on remplace les NaN, puis en str
            names_for_rows = names_for_rows.astype(object)
            names_for_rows = names_for_rows.where(names_for_rows.notna(), "Inconnu").astype(str)
        else:
            names_for_rows = data_mca.index.astype(str)

        # =========================
        # ACM avec prince
        # =========================
        mca = prince.MCA(
            n_components=2,
            random_state=42,
        )
        mca = mca.fit(data_mca)

        modalities_coords = mca.column_coordinates(data_mca)

        # On identifie correctement la variable et la modalité, même si le nom contient des "_"
        vars_list = mca_variables  # ["Kata", "N_Tour", "Victoire_norm"]
        variables = []
        modalites = []
        for idx in modalities_coords.index:
            var_name, mod = split_var_mod(idx, vars_list)
            variables.append(var_name)
            modalites.append(mod)

        modalities_coords["Variable"] = variables
        modalities_coords["Modalité"] = modalites

        # Coordonnées des individus (optionnel)
        if display_individuals:
            individuals_coords = mca.row_coordinates(data_mca)

        # Inertie expliquée
        eigenvalues = mca.eigenvalues_
        total_inertia = eigenvalues.sum()
        explained_inertia = eigenvalues / total_inertia

        # =========================
        # Graphique ACM
        # =========================
        st.subheader("Représentation de l'ACM")

        fig = go.Figure()

        # Modalités
        if display_modalities:
            for var in mca_variables:
                label = var_labels[var]
                var_coords = modalities_coords[modalities_coords["Variable"] == var]
                fig.add_trace(
                    go.Scatter(
                        x=var_coords[0],
                        y=var_coords[1],
                        mode="markers+text",
                        name=f"Modalités de {label}",
                        text=var_coords["Modalité"],
                        textposition="top center",
                        marker=dict(size=10),
                    )
                )

        # Individus
        if display_individuals:
            fig.add_trace(
                go.Scatter(
                    x=individuals_coords[0],
                    y=individuals_coords[1],
                    mode="markers",
                    name="Individus",
                    marker=dict(size=5, color="grey", opacity=0.5),
                    text=names_for_rows,
                    hoverinfo="text",
                )
            )

        # Limites d’axes
        if display_individuals:
            x_coords = pd.concat([modalities_coords[0], individuals_coords[0]])
            y_coords = pd.concat([modalities_coords[1], individuals_coords[1]])
        else:
            x_coords = modalities_coords[0]
            y_coords = modalities_coords[1]

        x_min = x_coords.min() - 0.5
        x_max = x_coords.max() + 0.5
        y_min = y_coords.min() - 0.5
        y_max = y_coords.max() + 0.5

        fig.update_layout(
            title="Carte factorielle de l'ACM",
            xaxis_title=f"Dimension 1 ({explained_inertia[0] * 100:.2f}% d'inertie)",
            yaxis_title=f"Dimension 2 ({explained_inertia[1] * 100:.2f}% d'inertie)",
            showlegend=True,
            width=900,
            height=650,
        )

        fig.update_xaxes(range=[x_min, x_max], zeroline=False)
        fig.update_yaxes(range=[y_min, y_max], zeroline=False)

        # Axes à 0
        fig.add_shape(
            type="line",
            x0=x_min,
            y0=0,
            x1=x_max,
            y1=0,
            line=dict(color="black", width=1),
        )
        fig.add_shape(
            type="line",
            x0=0,
            y0=y_min,
            x1=0,
            y1=y_max,
            line=dict(color="black", width=1),
        )

        st.plotly_chart(fig, width="stretch")

        # =========================
        # Interprétation détaillée / pédagogique
        # =========================
        st.subheader("Interprétation de l'ACM")

        if st.button("Interpréter automatiquement l'ACM", key="acm_interpret"):
            # ---- Fonction d'interprétation détaillée
            def interpret_acm(mca_obj, data_mca_local, mca_vars):
                output = StringIO()

                # 1. Inertie / variance expliquée
                eigen = mca_obj.eigenvalues_
                total_inertie = eigen.sum()
                explained = eigen / total_inertie

                output.write("### 1. Variance expliquée par les dimensions\n\n")
                for i, eig in enumerate(explained):
                    var_pct = eig * 100
                    output.write(f"- Dimension {i+1} : **{var_pct:.2f}%** de la variance expliquée.\n")

                output.write("\n---\n")
                output.write("### 2. Association des katas avec la victoire / défaite\n\n")

                # Coordonnées des modalités
                coords = mca_obj.column_coordinates(data_mca_local)
                coords.columns = [f"Dim_{i+1}" for i in range(coords.shape[1])]

                # On reconstruit correctement Variable / Modalité
                vars_list = mca_vars  # ["Kata", "N_Tour", "Victoire_norm"]
                variables = []
                modalites = []
                index_raw = []

                for idx in coords.index:
                    var_name, mod = split_var_mod(idx, vars_list)
                    variables.append(var_name)
                    modalites.append(mod)
                    index_raw.append(str(idx))

                coords["Variable"] = variables
                coords["Modalité"] = modalites
                coords["Index_raw"] = index_raw  # <== nouveau

                kata_coords = coords[coords["Variable"] == "Kata"].copy()
                victoire_coords = coords[coords["Variable"] == "Victoire_norm"].copy()

                if victoire_coords.empty:
                    output.write(
                        "Impossible d'identifier les modalités de la variable `Victoire` "
                        "dans cette configuration (après filtrage et normalisation).\n\n"
                    )
                else:
                    # On rend la détection de True / False très tolérante :
                    # - on regarde les MODALITÉS
                    # - ET le nom de colonne brut (Index_raw)
                    mods_raw = victoire_coords["Modalité"].astype(str).str.strip().str.lower()
                    idx_raw = victoire_coords["Index_raw"].astype(str).str.strip().str.lower()

                    true_like = {"true", "1", "vrai", "oui", "y", "o", "yes"}
                    false_like = {"false", "0", "faux", "non", "n", "no"}

                    is_true = mods_raw.isin(true_like) | idx_raw.str.contains("true")
                    is_false = mods_raw.isin(false_like) | idx_raw.str.contains("false")

                    victoire_coords["is_true"] = is_true
                    victoire_coords["is_false"] = is_false

                    vic_true = victoire_coords[victoire_coords["is_true"]]
                    vic_false = victoire_coords[victoire_coords["is_false"]]

                    if not vic_true.empty and not vic_false.empty:
                        # ✅ Cas idéal : au moins une modalité True ET une False
                        victoire_true = vic_true.iloc[0]
                        victoire_false = vic_false.iloc[0]

                        kata_coords["Distance_to_Victoire_True"] = np.sqrt(
                            (kata_coords["Dim_1"] - victoire_true["Dim_1"]) ** 2
                            + (kata_coords["Dim_2"] - victoire_true["Dim_2"]) ** 2
                        )
                        kata_coords["Distance_to_Victoire_False"] = np.sqrt(
                            (kata_coords["Dim_1"] - victoire_false["Dim_1"]) ** 2
                            + (kata_coords["Dim_2"] - victoire_false["Dim_2"]) ** 2
                        )

                        kata_coords["Proximity"] = (
                            kata_coords["Distance_to_Victoire_False"]
                            - kata_coords["Distance_to_Victoire_True"]
                        )

                        kata_coords = kata_coords.sort_values("Proximity", ascending=True)

                        output.write(
                            "Les katas les plus proches de **Victoire = True** "
                            "(potentiellement associés aux victoires) :\n\n"
                        )
                        top_win = kata_coords.head(5)
                        if top_win.empty:
                            output.write("- Aucun kata interprétable dans cette configuration.\n")
                        else:
                            for _, row in top_win.iterrows():
                                output.write(
                                    f"- **{row['Modalité']}** "
                                    f"(proximité = {row['Proximity']:.2f})\n"
                                )

                        output.write(
                            "\nLes katas les plus proches de **Victoire = False** "
                            "(potentiellement associés aux défaites) :\n\n"
                        )
                        top_lose = kata_coords.tail(5)
                        if top_lose.empty:
                            output.write("- Aucun kata interprétable dans cette configuration.\n")
                        else:
                            for _, row in top_lose.iterrows():
                                output.write(
                                    f"- **{row['Modalité']}** "
                                    f"(proximité = {row['Proximity']:.2f})\n"
                                )

                    elif not vic_true.empty and vic_false.empty:
                        # 🔹 Tous (ou quasi tous) les combats sont gagnés
                        output.write(
                            "Avec les filtres actuels, les combats observés sont quasiment "
                            "tous du côté **`Victoire = True`** (aucune modalité claire `False`).\n\n"
                        )
                        output.write(
                            "On peut tout de même lister les katas les plus proches "
                            "de cette modalité de victoire :\n\n"
                        )

                        victoire_true = vic_true.iloc[0]
                        kata_coords["Distance_to_Victoire_True"] = np.sqrt(
                            (kata_coords["Dim_1"] - victoire_true["Dim_1"]) ** 2
                            + (kata_coords["Dim_2"] - victoire_true["Dim_2"]) ** 2
                        )
                        kata_coords = kata_coords.sort_values("Distance_to_Victoire_True", ascending=True)

                        top_win = kata_coords.head(5)
                        if top_win.empty:
                            output.write("- Aucun kata interprétable dans cette configuration.\n")
                        else:
                            for _, row in top_win.iterrows():
                                output.write(
                                    f"- **{row['Modalité']}** "
                                    f"(proche de Victoire = True)\n"
                                )

                    elif not vic_false.empty and vic_true.empty:
                        # 🔹 Tous (ou quasi tous) les combats sont perdus
                        output.write(
                            "Avec les filtres actuels, les combats observés sont quasiment "
                            "tous du côté **`Victoire = False`** (aucune modalité claire `True`).\n\n"
                        )
                        output.write(
                            "On peut tout de même lister les katas les plus proches "
                            "de cette modalité de défaite :\n\n"
                        )

                        victoire_false = vic_false.iloc[0]
                        kata_coords["Distance_to_Victoire_False"] = np.sqrt(
                            (kata_coords["Dim_1"] - victoire_false["Dim_1"]) ** 2
                            + (kata_coords["Dim_2"] - victoire_false["Dim_2"]) ** 2
                        )
                        kata_coords = kata_coords.sort_values("Distance_to_Victoire_False", ascending=True)

                        top_lose = kata_coords.head(5)
                        if top_lose.empty:
                            output.write("- Aucun kata interprétable dans cette configuration.\n")
                        else:
                            for _, row in top_lose.iterrows():
                                output.write(
                                    f"- **{row['Modalité']}** "
                                    f"(proche de Victoire = False)\n"
                                )
                    else:
                        # 🔹 Cas vraiment tordu : impossible de repérer True/False,
                        # on revient au plan B (extrêmes sur Dim 1)
                        output.write(
                            "Les modalités de la variable `Victoire` ne peuvent pas être "
                            "reliées clairement à `True` ou `False` (codes non standards ou "
                            "données extrêmement filtrées).\n\n"
                        )
                        output.write(
                            "On peut toutefois repérer les katas les plus **extrêmes sur la "
                            "Dimension 1** de l'ACM (ce sont ceux qui contribuent le plus à "
                            "l'opposition principale observée) :\n\n"
                        )
                        if not kata_coords.empty:
                            kata_ext = kata_coords.copy()
                            kata_ext["Abs_Dim1"] = kata_ext["Dim_1"].abs()
                            kata_ext = kata_ext.sort_values("Abs_Dim1", ascending=False)
                            top_ext = kata_ext.head(5)
                            for _, row in top_ext.iterrows():
                                sens = "côté positif" if row["Dim_1"] >= 0 else "côté négatif"
                                output.write(
                                    f"- **{row['Modalité']}** "
                                    f"(fortement positionné sur la Dimension 1, {sens}).\n"
                                )
                        else:
                            output.write("- Aucun kata interprétable dans cette configuration.\n")

                output.write("\n---\n")
                output.write("### 3. Comment lire ces résultats (version vulgarisée)\n\n")
                output.write(
                    "- Chaque **point** sur le graphique représente soit une **modalité** "
                    "(par ex. un kata, un tour, True/False pour la victoire), soit un **combat** "
                    "(si les individus sont affichés).\n"
                )
                output.write(
                    "- Quand deux modalités sont **proches**, cela signifie qu'elles apparaissent "
                    "**souvent ensemble** dans les mêmes combats.\n"
                )
                output.write(
                    "- Par exemple, si un kata est positionné près de la modalité "
                    "**`Victoire = True`**, cela suggère que ce kata est plus souvent associé "
                    "à des combats **gagnés**.\n"
                )
                output.write(
                    "- À l'inverse, s'il est situé près de **`Victoire = False`**, il semble "
                    "davantage associé à des **défaites**.\n\n"
                )
                output.write(
                    "> ⚠️ Attention : l'ACM met en évidence des **associations statistiques**, "
                    "pas une relation de cause à effet. Un kata proche de `Victoire = True` ne "
                    "fait pas \"gagner\" à lui seul, il est simplement **fréquent** dans les combats gagnés.\n"
                )

                return output.getvalue()

            interpretation = interpret_acm(mca, data_mca, mca_variables)
            st.markdown(interpretation)

        else:
            # Explication statique
            st.markdown(
                """
                L’**analyse des correspondances multiples (ACM)** sert à explorer les relations entre plusieurs variables
                **qualitatives** (ici : *Kata*, *N_Tour*, *Victoire*).

                ### Comment lire le graphique

                - Chaque **modalité** (par ex. un nom de kata, un tour, `True/False` pour la victoire) est représentée par un point.
                - Les modalités **proches** sur le plan factoriel ont tendance à apparaître **ensemble** dans les mêmes combats.
                - La **Dimension 1** (axe horizontal) et la **Dimension 2** (axe vertical) sont les deux directions qui résument le mieux
                  la variabilité des données.
                - Les modalités **`Victoire = True`** et **`Victoire = False`** servent de repères :
                    - Les katas proches de **`Victoire = True`** sont plutôt associés à des combats gagnés.
                    - Les katas proches de **`Victoire = False`** sont plutôt associés à des combats perdus.

                ### À garder en tête

                - L’ACM montre des **associations**, pas une causalité directe.
                - Les résultats dépendent fortement des **filtres** choisis à gauche (type de compétition, types de tours, sexe, style, katas).
                - Pour affiner l’analyse, tu peux :
                    - restreindre à un seul sexe,
                    - ne garder que certains types de tours,
                    - filtrer par style (Shotokan / Shitoryu),
                    - exclure des katas très peu fréquents, etc.

                Tu peux ensuite cliquer sur **“Interpréter automatiquement l’ACM”** pour obtenir une lecture guidée des katas
                les plus associés aux victoires et aux défaites.
                """
            )
