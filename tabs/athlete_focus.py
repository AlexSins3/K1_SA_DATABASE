# tabs/athlete_focus.py

from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def show_athlete_focus_tab(data: pd.DataFrame) -> None:
    st.header("Focus Athlète")

    df = data.copy()

    # =========================
    # Mise en page : bande de filtres à gauche
    # =========================
    filters_col, content_col = st.columns([0.9, 2.4])

    # On prépare toutes les variables dont on aura besoin plus tard
    data_athlete = df.copy()
    selected_type_compet = None
    selected_sexe = None
    selected_athlete = None
    selected_compare_athlete = "Aucun"
    athlete_data = pd.DataFrame()
    compare_athlete_data = None
    selected_tours = []
    selected_competitions = []

    # =========================
    # Colonne de gauche : filtres
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
        st.markdown("### 🎯 Filtres")

        # ---- Type de compétition
        type_compet_options = ["Tous", "Premier League (K1)", "Series A (SA)"]
        selected_type_compet = st.radio(
            "Type de compétition",
            type_compet_options,
            key="athlete_type_compet",
        )

        if selected_type_compet != "Tous":
            if selected_type_compet == "Premier League (K1)":
                data_athlete = df[df["Type_Compet"] == "K1"].copy()
            elif selected_type_compet == "Series A (SA)":
                data_athlete = df[df["Type_Compet"] == "SA"].copy()
        else:
            data_athlete = df.copy()

        # ---- Sexe
        sexe_options = data_athlete["Sexe"].dropna().unique().tolist()
        sexe_options = sorted(sexe_options)
        if not sexe_options:
            st.warning("Aucun sexe disponible dans les données filtrées.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        selected_sexe = st.radio(
            "Sexe des athlètes",
            sexe_options,
            key="athlete_sexe",
        )

        data_athlete = data_athlete[data_athlete["Sexe"] == selected_sexe].copy()

        # ---- Liste des athlètes
        athlete_names = data_athlete["Nom"].dropna().unique().tolist()
        athlete_names.sort()

        if not athlete_names:
            st.warning("Aucun athlète disponible avec les filtres sélectionnés.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        selected_athlete = st.selectbox(
            "Athlète principal", athlete_names, key="athlete_main_select"
        )

        athlete_data = data_athlete[data_athlete["Nom"] == selected_athlete].copy()

        # ---- Athlète comparé
        compare_options = ["Aucun"] + [name for name in athlete_names if name != selected_athlete]
        selected_compare_athlete = st.selectbox(
            "Comparer à", compare_options, key="athlete_compare_select"
        )

        if selected_compare_athlete != "Aucun":
            compare_athlete_data = data_athlete[data_athlete["Nom"] == selected_compare_athlete].copy()
        else:
            compare_athlete_data = None

        st.markdown("---")
        st.markdown("#### 🔎 Filtres avancés")

        # ---- Filtres pour les tours (histogrammes Kata)
        if compare_athlete_data is not None:
            tour_options = sorted(
                set(athlete_data["N_Tour"].dropna().unique())
                | set(compare_athlete_data["N_Tour"].dropna().unique())
            )
        else:
            tour_options = sorted(athlete_data["N_Tour"].dropna().unique().tolist())

        if tour_options:
            selected_tours = st.multiselect(
                "Tours (N_Tour)",
                options=tour_options,
                default=tour_options,
                key="athlete_tours_filter",
            )
        else:
            selected_tours = []

        # ---- Filtres pour les compétitions (Kiviat)
        if compare_athlete_data is not None:
            competition_options = sorted(
                set(athlete_data["Competition"].dropna().unique())
                | set(compare_athlete_data["Competition"].dropna().unique())
            )
        else:
            competition_options = sorted(athlete_data["Competition"].dropna().unique().tolist())

        if competition_options:
            selected_competitions = st.multiselect(
                "Compétitions",
                options=competition_options,
                default=competition_options,
                key="athlete_compet_filter",
            )
        else:
            selected_competitions = []

        st.markdown("</div>", unsafe_allow_html=True)

    # =========================
    # Helpers pour les tours / compétitions
    # =========================

    # Colonne d'année : on force "Year" si présente
    year_col = "Year" if "Year" in df.columns else None

    def build_compet_label(row: pd.Series) -> str:
        """Construit un label Competition + année si disponible."""
        comp = str(row.get("Competition", ""))
        if year_col is not None:
            year_val = row.get(year_col)
            if pd.notna(year_val):
                try:
                    year_int = int(year_val)
                    return f"{comp}_{year_int}"
                except Exception:
                    return comp
        return comp

    # Mapping des tours pour K1 (tour max par compétition)
    K1_TOUR_LABEL_FROM_ROW = {
        "Pool_1": "Poule",
        "Pool_2": "Poule",
        "Pool_3": "Poule",
        "R1": "Quart de finale",
        "R2": "Demi finale",
        "Finale": "Finale",
        "Final": "Finale",
    }

    K1_TOUR_ORDER = {
        "Poule": 1,
        "Quart de finale": 2,
        "Demi finale": 3,
        "Place de 3": 4,
        "Bronze": 4,  # même niveau, issue différente
        "Finale": 5,
    }

    # Mapping des tours pour SA (tour max par compétition)
    SA_TOUR_LABEL_FROM_ROW = {
        "T1": "1er Tour",
        "T2": "2ème Tour",
        "T3": "3ème Tour",
        "R1": "Quart de finale",
        "R2": "Demi finale",
        "Finale": "Finale",
        "Final": "Finale",
        "PW1": "Finale de poule",
        "PW2": "Finale de poule",
        "PW3": "Finale de poule",
    }

    SA_TOUR_ORDER = {
        "1er Tour": 1,
        "2ème Tour": 2,
        "3ème Tour": 3,
        "Finale de poule": 4,
        "Quart de finale": 5,
        "Demi finale": 6,
        "Place de 3": 7,
        "Bronze": 7,
        "Finale": 8,
    }

    def map_k1_tour(row: pd.Series):
        n_tour = row.get("N_Tour")
        victoire = row.get("Victoire")
        if pd.isna(n_tour):
            return None
        n_tour = str(n_tour)

        if n_tour == "Bronze":
            # différenciation Bronze / place de 3
            if pd.notna(victoire) and bool(victoire):
                return "Bronze"
            else:
                return "Place de 3"

        return K1_TOUR_LABEL_FROM_ROW.get(n_tour)

    def map_sa_tour(row: pd.Series):
        n_tour = row.get("N_Tour")
        victoire = row.get("Victoire")
        if pd.isna(n_tour):
            return None
        n_tour = str(n_tour)

        if n_tour == "Bronze":
            if pd.notna(victoire) and bool(victoire):
                return "Bronze"
            else:
                return "Place de 3"

        return SA_TOUR_LABEL_FROM_ROW.get(n_tour)

    # Mapping commun pour le Kiviat "moyenne des notes par tour"
    KIVIAT_TOUR_MAP = {
        "Pool_1": "Poule",
        "Pool_2": "Poule",
        "Pool_3": "Poule",
        "R1": "Quart de finale",
        "R2": "Demi finale",
        "Bronze": "Bronze / Place de 3",
        "Finale": "Finale",
        "Final": "Finale",
        "T1": "1er Tour",
        "T2": "2ème Tour",
        "T3": "3ème Tour",
        "PW1": "Finale de poule",
        "PW2": "Finale de poule",
        "PW3": "Finale de poule",
    }

    KIVIAT_TOUR_ORDER = {
        "Poule": 1,
        "1er Tour": 2,
        "2ème Tour": 3,
        "3ème Tour": 4,
        "Finale de poule": 5,
        "Quart de finale": 6,
        "Demi finale": 7,
        "Bronze / Place de 3": 8,
        "Finale": 9,
    }

    def map_tour_for_kiviat(row: pd.Series):
        n_tour = row.get("N_Tour")
        if pd.isna(n_tour):
            return None
        n_tour = str(n_tour)
        return KIVIAT_TOUR_MAP.get(n_tour)

    def athlete_label_html(name: str) -> str:
        return f"<p style='font-size:13px;font-style:italic;color:#555;'>Athlète : {name}</p>"

    def highlight_victory_series(s: pd.Series):
        styles = []
        for v in s:
            if v == "Oui":
                styles.append("background-color: #d4edda;")  # vert clair
            elif v == "Non":
                styles.append("background-color: #f8d7da;")  # rouge clair
            else:
                styles.append("")
        return styles

    def victoire_to_str(v):
        """Convertit la valeur brute de 'Victoire' en 'Oui' / 'Non' / 'Inconnu' de façon robuste."""
        if pd.isna(v):
            return "Inconnu"

        # Bool pur
        if isinstance(v, (bool, np.bool_)):
            return "Oui" if v else "Non"

        # Numérique (0 / 1)
        try:
            nv = float(v)
            if nv == 1:
                return "Oui"
            if nv == 0:
                return "Non"
        except Exception:
            pass

        # Texte
        s = str(v).strip().lower()
        if s in {"True", "true", "vrai", "yes", "oui", "win", "gagné", "gagne", "1"}:
            return "Oui"
        if s in {"False", "false", "faux", "no", "non", "lose", "perdu", "0"}:
            return "Non"

        # Fallback (au cas où)
        return "Oui" if bool(v) else "Non"

    # =========================
    # Colonne de droite : contenu / visualisations
    # =========================
    with content_col:
        if athlete_data.empty:
            st.warning("Aucune donnée pour l'athlète sélectionné.")
            return

        # ==============
        # 1. Info athlètes
        # ==============
        st.subheader("Informations Athlète(s)")

        col_left, col_separator, col_right = st.columns([1, 0.05, 1])

        # --- Athlète principal
        with col_left:
            st.markdown(athlete_label_html(selected_athlete), unsafe_allow_html=True)

            sexe = athlete_data["Sexe"].mode()[0] if not athlete_data["Sexe"].mode().empty else "Non spécifié"
            nation = athlete_data["Nation"].mode()[0] if not athlete_data["Nation"].mode().empty else "Non spécifié"
            style = athlete_data["Style"].mode()[0] if not athlete_data["Style"].mode().empty else "Non spécifié"

            # Dernier âge connu
            age_series = athlete_data["Age"].dropna()
            if age_series.empty:
                age_display = "Non spécifié"
            else:
                last_age = age_series.iloc[-1]
                try:
                    age_display = f"{float(last_age):.1f} ans"
                except Exception:
                    age_display = f"{last_age} ans"

            # Dernier ranking connu
            ranking_series = athlete_data["Ranking"].dropna()
            if ranking_series.empty:
                ranking_str = "Non spécifié"
            else:
                last_rank = ranking_series.iloc[-1]
                try:
                    last_rank_int = int(last_rank)
                    ranking_str = f"{last_rank_int}"
                except Exception:
                    ranking_str = f"{last_rank}"

            st.markdown(
                f"""
                - **Sexe :** {sexe}  
                - **Âge (dernier connu) :** {age_display}  
                - **Ranking (dernier connu) :** {ranking_str}  
                - **Nationalité :** {nation}  
                - **Style :** {style}
                """
            )

        with col_separator:
            if compare_athlete_data is not None and not compare_athlete_data.empty:
                st.markdown(
                    "<div style='border-left:1px solid #d0d0d0;height:100%;'></div>",
                    unsafe_allow_html=True,
                )

        # --- Athlète comparé
        with col_right:
            if compare_athlete_data is not None and not compare_athlete_data.empty:
                st.markdown(
                    athlete_label_html(selected_compare_athlete),
                    unsafe_allow_html=True,
                )

                sexe_comp = (
                    compare_athlete_data["Sexe"].mode()[0]
                    if not compare_athlete_data["Sexe"].mode().empty
                    else "Non spécifié"
                )
                nation_comp = (
                    compare_athlete_data["Nation"].mode()[0]
                    if not compare_athlete_data["Nation"].mode().empty
                    else "Non spécifié"
                )
                style_comp = (
                    compare_athlete_data["Style"].mode()[0]
                    if not compare_athlete_data["Style"].mode().empty
                    else "Non spécifié"
                )

                # Dernier âge connu (comparé)
                age_series_comp = compare_athlete_data["Age"].dropna()
                if age_series_comp.empty:
                    age_comp_display = "Non spécifié"
                else:
                    last_age_comp = age_series_comp.iloc[-1]
                    try:
                        age_comp_display = f"{float(last_age_comp):.1f} ans"
                    except Exception:
                        age_comp_display = f"{last_age_comp} ans"

                # Dernier ranking connu (comparé)
                ranking_series_comp = compare_athlete_data["Ranking"].dropna()
                if ranking_series_comp.empty:
                    ranking_str_comp = "Non spécifié"
                else:
                    last_rank_comp = ranking_series_comp.iloc[-1]
                    try:
                        last_rank_comp_int = int(last_rank_comp)
                        ranking_str_comp = f"{last_rank_comp_int}"
                    except Exception:
                        ranking_str_comp = f"{last_rank_comp}"

                st.markdown(
                    f"""
                    - **Sexe :** {sexe_comp}  
                    - **Âge (dernier connu) :** {age_comp_display}  
                    - **Ranking (dernier connu) :** {ranking_str_comp}  
                    - **Nationalité :** {nation_comp}  
                    - **Style :** {style_comp}
                    """
                )
            else:
                st.markdown(
                    "<p style='font-size:12px;color:gray;'>Aucun athlète comparé sélectionné</p>",
                    unsafe_allow_html=True,
                )

        # ==============
        # 2. Tour maximal atteint par compétition (K1 vs SA)
        # ==============
        st.subheader("Tour maximal atteint par compétition")

        col_left, col_separator, col_right = st.columns([1, 0.05, 1])

        # Préparation des données pour K1 et SA pour chacun des athlètes
        def compute_max_tour_by_compet(df_src: pd.DataFrame, athlete_name: str, type_compet: str) -> pd.DataFrame:
            """Retourne un DataFrame avec Competition_Label, Tour_Label, Level, Athlète pour un type de compétition."""
            df_a = df_src[(df_src["Nom"] == athlete_name) & (df_src["Type_Compet"] == type_compet)].copy()
            if df_a.empty:
                return pd.DataFrame(columns=["Competition_Label", "Tour_Label", "Level", "Athlète"])

            # Construction du label compétition + année
            df_a["Competition_Label"] = df_a.apply(build_compet_label, axis=1)

            # Ajout d'une colonne numérique d'année pour le tri
            if year_col is not None and year_col in df_a.columns:
                df_a["Year_numeric"] = pd.to_numeric(df_a[year_col], errors="coerce")
            else:
                df_a["Year_numeric"] = np.nan

            if type_compet == "K1":
                df_a["Tour_Label"] = df_a.apply(map_k1_tour, axis=1)
                df_a["Level"] = df_a["Tour_Label"].map(K1_TOUR_ORDER)
            else:  # SA
                df_a["Tour_Label"] = df_a.apply(map_sa_tour, axis=1)
                df_a["Level"] = df_a["Tour_Label"].map(SA_TOUR_ORDER)

            df_a = df_a.dropna(subset=["Competition_Label", "Tour_Label", "Level"])

            if df_a.empty:
                return pd.DataFrame(columns=["Competition_Label", "Tour_Label", "Level", "Athlète"])

            # On garde, pour chaque compétition, le tour maximal (niveau le plus élevé)
            idx = df_a.groupby("Competition_Label")["Level"].idxmax()
            out = df_a.loc[idx, ["Competition_Label", "Tour_Label", "Level", "Year_numeric"]].copy()
            out["Athlète"] = athlete_name

            # Tri chronologique (année) puis par nom de compète
            out = out.sort_values(["Year_numeric", "Competition_Label"], na_position="first")
            return out

        # K1
        df_k1 = compute_max_tour_by_compet(df, selected_athlete, "K1")
        if compare_athlete_data is not None and not compare_athlete_data.empty:
            df_k1_comp = compute_max_tour_by_compet(df, selected_compare_athlete, "K1")
            df_k1 = pd.concat([df_k1, df_k1_comp], ignore_index=True)

        # SA
        df_sa = compute_max_tour_by_compet(df, selected_athlete, "SA")
        if compare_athlete_data is not None and not compare_athlete_data.empty:
            df_sa_comp = compute_max_tour_by_compet(df, selected_compare_athlete, "SA")
            df_sa = pd.concat([df_sa, df_sa_comp], ignore_index=True)

        # Graph K1 (gauche)
        with col_left:
            st.markdown("### Premier League (K1)")
            if df_k1.empty:
                st.info("Aucune donnée K1 pour les athlètes sélectionnés.")
            else:
                # Ordre des catégories selon tri chronologique
                cat_order_k1 = df_k1.sort_values(
                    ["Year_numeric", "Competition_Label"], na_position="first"
                )["Competition_Label"].unique()

                unique_levels = sorted(df_k1["Level"].unique())
                tickvals = unique_levels
                ticktexts = []
                for lvl in unique_levels:
                    labels_at_level = sorted(df_k1.loc[df_k1["Level"] == lvl, "Tour_Label"].unique())
                    ticktexts.append(" / ".join(labels_at_level))

                fig_k1 = px.bar(
                    df_k1,
                    x="Competition_Label",
                    y="Level",
                    color="Athlète",
                    barmode="group",
                    text="Tour_Label",
                    labels={"Competition_Label": "Compétition", "Level": "Tour maximal"},
                    category_orders={"Competition_Label": list(cat_order_k1)},
                )
                fig_k1.update_traces(textposition="outside")
                fig_k1.update_yaxes(tickmode="array", tickvals=tickvals, ticktext=ticktexts)

                st.plotly_chart(
                    fig_k1,
                    width="stretch",
                    key=f"max_tour_k1_{selected_sexe}_{selected_type_compet}_{selected_athlete}_{selected_compare_athlete}",
                )

        with col_separator:
            st.markdown(
                "<div style='border-left:1px solid #d0d0d0;height:100%;'></div>",
                unsafe_allow_html=True,
            )

        # Graph SA (droite)
        with col_right:
            st.markdown("### Series A (SA)")
            if df_sa.empty:
                st.info("Aucune donnée SA pour les athlètes sélectionnés.")
            else:
                cat_order_sa = df_sa.sort_values(
                    ["Year_numeric", "Competition_Label"], na_position="first"
                )["Competition_Label"].unique()

                unique_levels_sa = sorted(df_sa["Level"].unique())
                tickvals_sa = unique_levels_sa
                ticktexts_sa = []
                for lvl in unique_levels_sa:
                    labels_at_level = sorted(df_sa.loc[df_sa["Level"] == lvl, "Tour_Label"].unique())
                    ticktexts_sa.append(" / ".join(labels_at_level))

                fig_sa = px.bar(
                    df_sa,
                    x="Competition_Label",
                    y="Level",
                    color="Athlète",
                    barmode="group",
                    text="Tour_Label",
                    labels={"Competition_Label": "Compétition", "Level": "Tour maximal"},
                    category_orders={"Competition_Label": list(cat_order_sa)},
                )
                fig_sa.update_traces(textposition="outside")
                fig_sa.update_yaxes(tickmode="array", tickvals=tickvals_sa, ticktext=ticktexts_sa)

                st.plotly_chart(
                    fig_sa,
                    width="stretch",
                    key=f"max_tour_sa_{selected_sexe}_{selected_type_compet}_{selected_athlete}_{selected_compare_athlete}",
                )

        # ==============
        # 3. Histogramme des Katas effectués
        # ==============
        st.subheader("Histogramme des Katas effectués")

        col_left, col_separator, col_right = st.columns([1, 0.05, 1])

        # Athlète principal
        with col_left:
            st.markdown(athlete_label_html(selected_athlete), unsafe_allow_html=True)
            if selected_tours:
                kata_data = athlete_data[athlete_data["N_Tour"].isin(selected_tours)]
            else:
                kata_data = athlete_data.copy()

            kata_counts = kata_data["Kata"].value_counts().reset_index()
            kata_counts.columns = ["Kata", "Nombre"]
            kata_counts = kata_counts[kata_counts["Nombre"] > 0]

            if kata_counts.empty:
                st.warning("Aucun Kata à afficher pour les tours sélectionnés.")
            else:
                fig_kata = px.bar(
                    kata_counts,
                    x="Kata",
                    y="Nombre",
                    title="Nombre de Katas effectués",
                    labels={"Nombre": "Nombre de fois"},
                    text="Nombre",
                )
                fig_kata.update_layout(xaxis_title="Kata", yaxis_title="Nombre de fois")
                fig_kata.update_traces(textposition="outside")

                st.plotly_chart(
                    fig_kata,
                    width="stretch",
                    key=f"hist_kata_main_{selected_sexe}_{selected_type_compet}_{selected_athlete}_{len(selected_tours)}",
                )

        with col_separator:
            if compare_athlete_data is not None and not compare_athlete_data.empty:
                st.markdown(
                    "<div style='border-left:1px solid #d0d0d0;height:100%;'></div>",
                    unsafe_allow_html=True,
                )

        # Athlète comparé
        with col_right:
            if compare_athlete_data is not None and not compare_athlete_data.empty:
                st.markdown(
                    athlete_label_html(selected_compare_athlete),
                    unsafe_allow_html=True,
                )
                if selected_tours:
                    kata_data_comp = compare_athlete_data[compare_athlete_data["N_Tour"].isin(selected_tours)]
                else:
                    kata_data_comp = compare_athlete_data.copy()

                kata_counts_comp = kata_data_comp["Kata"].value_counts().reset_index()
                kata_counts_comp.columns = ["Kata", "Nombre"]
                kata_counts_comp = kata_counts_comp[kata_counts_comp["Nombre"] > 0]

                if kata_counts_comp.empty:
                    st.warning("Aucun Kata à afficher pour les tours sélectionnés.")
                else:
                    fig_kata_comp = px.bar(
                        kata_counts_comp,
                        x="Kata",
                        y="Nombre",
                        title="Nombre de Katas effectués",
                        labels={"Nombre": "Nombre de fois"},
                        text="Nombre",
                    )
                    fig_kata_comp.update_layout(xaxis_title="Kata", yaxis_title="Nombre de fois")
                    fig_kata_comp.update_traces(textposition="outside")

                    st.plotly_chart(
                        fig_kata_comp,
                        width="stretch",
                        key=f"hist_kata_comp_{selected_sexe}_{selected_type_compet}_{selected_compare_athlete}_{selected_athlete}_{len(selected_tours)}",
                    )
            else:
                st.markdown(
                    "<p style='font-size:12px;color:gray;'>Aucun athlète comparé sélectionné</p>",
                    unsafe_allow_html=True,
                )

        # ==============
        # 4. Kiviat moyenne des notes par N_Tour (avec seulement les tours présents)
        # ==============
        st.subheader("Moyenne des notes par Tour")

        def compute_avg_notes_by_tour(df_source: pd.DataFrame, athlete_name: str) -> pd.DataFrame:
            if selected_competitions:
                note_data = df_source[df_source["Competition"].isin(selected_competitions)]
            else:
                note_data = df_source.copy()

            note_data = note_data.dropna(subset=["Note", "N_Tour"])

            if note_data.empty:
                return pd.DataFrame(columns=["Tour", "Moyenne_Note", "Athlète"])

            note_data["Tour_Kiviat"] = note_data.apply(map_tour_for_kiviat, axis=1)
            note_data = note_data.dropna(subset=["Tour_Kiviat"])

            if note_data.empty:
                return pd.DataFrame(columns=["Tour", "Moyenne_Note", "Athlète"])

            grouped = note_data.groupby("Tour_Kiviat")["Note"].mean().reset_index()
            grouped["Athlète"] = athlete_name

            # On ajoute un ordre pour organiser le Kiviat
            grouped["Order"] = grouped["Tour_Kiviat"].map(KIVIAT_TOUR_ORDER).fillna(999)
            grouped = grouped.sort_values("Order")

            grouped.rename(columns={"Tour_Kiviat": "Tour", "Note": "Moyenne_Note"}, inplace=True)
            return grouped[["Tour", "Moyenne_Note", "Athlète"]]

        df_kiviat_tour = compute_avg_notes_by_tour(athlete_data, selected_athlete)

        if compare_athlete_data is not None and not compare_athlete_data.empty:
            df_kiviat_tour_comp = compute_avg_notes_by_tour(compare_athlete_data, selected_compare_athlete)
            df_kiviat_tour = pd.concat([df_kiviat_tour, df_kiviat_tour_comp], ignore_index=True)

        if df_kiviat_tour.empty:
            st.info("Aucune note disponible pour construire le diagramme par tour.")
        else:
            fig_kiviat_tour = go.Figure()

            for athlete_name in df_kiviat_tour["Athlète"].unique():
                subset = df_kiviat_tour[df_kiviat_tour["Athlète"] == athlete_name]
                fig_kiviat_tour.add_trace(
                    go.Scatterpolar(
                        r=subset["Moyenne_Note"],
                        theta=subset["Tour"],
                        fill="toself",
                        name=athlete_name,
                    )
                )

            fig_kiviat_tour.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[30.0, 47.0],
                    )
                ),
                showlegend=True,
                title="Moyenne des notes par Tour (tours réellement disputés)",
            )

            st.plotly_chart(
                fig_kiviat_tour,
                width="stretch",
                key=f"kiviat_tour_{selected_sexe}_{selected_type_compet}_{selected_athlete}_{selected_compare_athlete}_{len(selected_competitions)}",
            )

        # ==============
        # 5. Kiviat moyenne des notes par Kata (hover kata + note)
        # ==============
        st.subheader("Moyenne des notes par Kata")

        def get_katas_for_athlete(df_source: pd.DataFrame) -> set:
            d = df_source.copy()
            if selected_competitions:
                d = d[d["Competition"].isin(selected_competitions)]
            return set(d["Kata"].dropna().unique())

        katas_a = get_katas_for_athlete(athlete_data)

        if compare_athlete_data is not None and not compare_athlete_data.empty:
            katas_b = get_katas_for_athlete(compare_athlete_data)
            common_katas = katas_a & katas_b
            if common_katas:
                kata_list = sorted(common_katas)
            else:
                kata_list = sorted(katas_a | katas_b)
        else:
            kata_list = sorted(katas_a)

        def compute_avg_notes_by_kata(df_source: pd.DataFrame, athlete_name: str) -> pd.DataFrame:
            d = df_source.copy()
            if selected_competitions:
                d = d[d["Competition"].isin(selected_competitions)]

            d = d.dropna(subset=["Note", "Kata"])
            if not kata_list:
                return pd.DataFrame(columns=["Kata", "Moyenne_Note", "Athlète"])

            d = d[d["Kata"].isin(kata_list)]
            if d.empty:
                return pd.DataFrame(columns=["Kata", "Moyenne_Note", "Athlète"])

            grouped = d.groupby("Kata")["Note"].mean().reset_index()
            grouped["Athlète"] = athlete_name
            grouped.rename(columns={"Note": "Moyenne_Note"}, inplace=True)
            return grouped[["Kata", "Moyenne_Note", "Athlète"]]

        df_kiviat_kata = compute_avg_notes_by_kata(athlete_data, selected_athlete)

        if compare_athlete_data is not None and not compare_athlete_data.empty:
            df_kiviat_kata_comp = compute_avg_notes_by_kata(compare_athlete_data, selected_compare_athlete)
            df_kiviat_kata = pd.concat([df_kiviat_kata, df_kiviat_kata_comp], ignore_index=True)

        if df_kiviat_kata.empty:
            st.info("Aucune note disponible pour construire le diagramme par Kata.")
        else:
            df_kiviat_kata["Kata"] = pd.Categorical(df_kiviat_kata["Kata"], categories=kata_list, ordered=True)
            df_kiviat_kata = df_kiviat_kata.sort_values("Kata")

            fig_kiviat_kata = go.Figure()

            for athlete_name in df_kiviat_kata["Athlète"].unique():
                subset = df_kiviat_kata[df_kiviat_kata["Athlète"] == athlete_name].copy()
                subset["note_hover"] = subset["Moyenne_Note"].astype(float).round(2)

                fig_kiviat_kata.add_trace(
                    go.Scatterpolar(
                        r=subset["Moyenne_Note"],
                        theta=subset["Kata"],
                        fill="toself",
                        name=athlete_name,
                        customdata=subset[["note_hover"]],
                        hovertemplate=(
                            "<b>%{theta}</b><br>"
                            "Note moyenne: %{customdata[0]:.2f}<br>"
                            "<extra>" + athlete_name + "</extra>"
                        ),
                    )
                )

            fig_kiviat_kata.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[30.0, 47.0],
                    )
                ),
                showlegend=True,
                title="Moyenne des notes par Kata",
            )

            st.plotly_chart(
                fig_kiviat_kata,
                width="stretch",
                key=f"kiviat_kata_{selected_sexe}_{selected_type_compet}_{selected_athlete}_{selected_compare_athlete}_{len(selected_competitions)}",
            )

        # ==============
        # 6. Historique : tours ou rencontres
        # ==============
        st.subheader("Historique")

        df_hist = df.copy()

        if selected_type_compet == "Premier League (K1)":
            df_hist = df_hist[df_hist["Type_Compet"] == "K1"]
        elif selected_type_compet == "Series A (SA)":
            df_hist = df_hist[df_hist["Type_Compet"] == "SA"]

        if selected_sexe is not None and "Sexe" in df_hist.columns:
            df_hist = df_hist[df_hist["Sexe"] == selected_sexe]

        if selected_competitions:
            df_hist = df_hist[df_hist["Competition"].isin(selected_competitions)]

        df_hist = df_hist.reset_index(drop=True)

        # Cas 1 : un seul athlète → Historique des tours
        if compare_athlete_data is None or compare_athlete_data.empty:
            st.markdown("##### Historique des tours")
            df_a = df_hist[df_hist["Nom"] == selected_athlete].copy()

            if df_a.empty:
                st.info("Aucun historique trouvé pour cet athlète avec les filtres actuels.")
            else:
                records = []

                for idx, row in df_a.iterrows():
                    ceinture = row.get("Ceinture")
                    opp_name = "Inconnu"

                    if ceinture == "R":
                        if idx + 1 < len(df_hist):
                            opp_row = df_hist.iloc[idx + 1]
                            if opp_row.get("Ceinture") == "B":
                                opp_name = opp_row.get("Nom", "Inconnu")
                    elif ceinture == "B":
                        if idx - 1 >= 0:
                            opp_row = df_hist.iloc[idx - 1]
                            if opp_row.get("Ceinture") == "R":
                                opp_name = opp_row.get("Nom", "Inconnu")

                    vic_val = row.get("Victoire")
                    vic_str = victoire_to_str(vic_val)

                    records.append(
                        {
                            "Tour": row.get("N_Tour"),
                            "Kata": row.get("Kata"),
                            "Note": row.get("Note"),
                            "vs.": opp_name,
                            "Victoire": vic_str,
                            "Competition": build_compet_label(row),
                        }
                    )

                hist_df = pd.DataFrame(records)
                hist_df = hist_df.sort_values(["Competition", "Tour"], na_position="last")

                styled = hist_df.style.apply(
                    lambda s: highlight_victory_series(s) if s.name == "Victoire" else [""] * len(s),
                    axis=0,
                )
                st.dataframe(styled, width="stretch")

        # Cas 2 : deux athlètes → Historique des rencontres
        else:
            st.markdown("##### Historique des rencontres")

            records = []
            n_rows = len(df_hist)

            for i in range(n_rows - 1):
                r1 = df_hist.iloc[i]
                r2 = df_hist.iloc[i + 1]

                noms = {r1.get("Nom"), r2.get("Nom")}
                ceintures = {str(r1.get("Ceinture")), str(r2.get("Ceinture"))}

                if {selected_athlete, selected_compare_athlete}.issubset(noms) and {"R", "B"}.issubset(ceintures):
                    if r1.get("Nom") == selected_athlete:
                        self_row = r1
                    else:
                        self_row = r2

                    vic_val = self_row.get("Victoire")
                    vic_str = victoire_to_str(vic_val)

                    records.append(
                        {
                            "Tour": self_row.get("N_Tour"),
                            "Kata": self_row.get("Kata"),
                            "Note": self_row.get("Note"),
                            "Ceinture": self_row.get("Ceinture"),
                            "Victoire": vic_str,
                            "Competition": build_compet_label(self_row),
                        }
                    )

            if not records:
                st.info("Les deux athlètes ne se sont pas encore affrontés (ou pas avec les filtres actuels).")
            else:
                meet_df = pd.DataFrame(records)
                meet_df = meet_df.sort_values(["Competition", "Tour"], na_position="last")

                styled_meet = meet_df.style.apply(
                    lambda s: highlight_victory_series(s) if s.name == "Victoire" else [""] * len(s),
                    axis=0,
                )
                st.dataframe(styled_meet, width="stretch")
