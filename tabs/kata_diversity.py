# tabs/kata_diversity.py — Onglet F : Analyse diversité kata
from __future__ import annotations

import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

from utils.ui import filter_panel_open, filter_panel_close
from utils.interpretations import show_tab_help, show_chart_guide, interpret_diversity
from utils.display import format_display_df, fmt_col
from utils.lang import t


def _tour_rank(tour) -> int:
    order = [
        "T1", "T2", "T3", "Pool_1", "Pool_2", "Pool_3",
        "PW1", "PW2", "PW3", "R1", "R2", "Bronze", "Final",
    ]
    try:
        if tour is None or pd.isna(tour):
            return 999
    except (TypeError, ValueError):
        pass
    try:
        return order.index(str(tour))
    except ValueError:
        return 999


@st.fragment
def show_kata_diversity_tab(data: pd.DataFrame) -> None:
    st.header(t("Analyse de la diversité kata"))
    show_tab_help("kata_diversity")

    df = data.copy()
    for col in ["Note", "Year"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Victoire" in df.columns:
        df["Victoire"] = df["Victoire"].astype(str).str.lower().isin(["true", "1", "vrai", "yes"]).astype(int)

    filters_col, content_col = st.columns([0.9, 2.4])

    with filters_col:
        filter_panel_open()

        type_opts = [t("Tous"), "K1", "SA"]
        sel_type = st.radio(t("Type compétition"), type_opts, key="div_type")
        d = df.copy()
        if sel_type != t("Tous"):
            d = d[d["Type_Compet"] == sel_type]

        sexes = sorted(d["Sexe"].dropna().unique().tolist())
        sel_sexe = st.selectbox(t("Sexe"), [t("Tous")] + sexes, key="div_sexe")
        if sel_sexe != t("Tous"):
            d = d[d["Sexe"] == sel_sexe]

        min_passages = st.slider(t("Min passages par athlète"), 3, 20, 5, key="div_min")

        filter_panel_close()

    with content_col:
        if d.empty:
            st.info(t("Aucune donnée."))
            return

        # ── Build athlete profile ──
        athlete_grp = d.groupby("Nom").agg(
            Katas_Distincts=("Kata", "nunique"),
            Passages=("Note", "count"),
            Note_Moy=("Note", "mean"),
            Note_Max=("Note", "max"),
            Victoires=("Victoire", "sum"),
            Total=("Victoire", "count"),
            Tour_Max=("N_Tour", lambda s: max(s.astype(str), key=_tour_rank) if len(s) > 0 else "—"),
        ).reset_index()
        athlete_grp["Win_Rate"] = (athlete_grp["Victoires"] / athlete_grp["Total"] * 100).round(1)
        athlete_grp["Note_Moy"] = athlete_grp["Note_Moy"].round(2)

        # Filter on min passages
        ag = athlete_grp[athlete_grp["Passages"] >= min_passages].copy()
        if ag.empty:
            st.warning(f"{t('Aucun athlète avec')} ≥ {min_passages} passages.")
            return

        # Classify diversity
        ag["Profil"] = pd.cut(
            ag["Katas_Distincts"],
            bins=[0, 2, 4, 100],
            labels=[t("Spécialiste (1-2)"), t("Modéré (3-4)"), t("Polyvalent (5+)")],
        )

        # ── KPIs ──
        c1, c2, c3 = st.columns(3)
        c1.metric(t("Athlètes analysés"), len(ag), help=t("Nombre d'athlètes ayant le minimum de passages requis"))
        c2.metric(t("Katas distincts moy."), f"{ag['Katas_Distincts'].mean():.1f}", help=t("En moyenne, combien de katas différents chaque athlète utilise"))
        c3.metric(t("Médiane katas"), int(ag["Katas_Distincts"].median()), help=t("La moitié des athlètes utilisent plus de katas, l'autre moitié moins"))

        # ── Scatter: diversity vs win rate ──
        st.subheader(t("Diversité kata vs Win rate"))
        show_chart_guide("scatter")
        fig_scatter = px.scatter(
            ag, x="Katas_Distincts", y="Win_Rate",
            color="Note_Moy", size="Passages",
            hover_data=["Nom", "Note_Moy", "Passages", "Tour_Max"],
            color_continuous_scale="RdYlGn",
            title=t("Nb katas distincts vs Win rate"),
            labels={"Katas_Distincts": t("Katas distincts"), "Win_Rate": "Win rate (%)"},
        )
        st.plotly_chart(fig_scatter, use_container_width=True, key="div_scatter")

        # ── Box: win rate by profil ──
        st.subheader(t("Win rate par profil de diversité"))
        show_chart_guide("boxplot")
        fig_box = px.box(
            ag, x="Profil", y="Win_Rate", color="Profil",
            title=t("Spécialistes vs Polyvalents – Win rate"),
            category_orders={"Profil": [t("Spécialiste (1-2)"), t("Modéré (3-4)"), t("Polyvalent (5+)")]},
            points="all",
        )
        st.plotly_chart(fig_box, use_container_width=True, key="div_box_wr")

        # ── Box: note by profil ──
        st.subheader(t("Note moyenne par profil de diversité"))
        fig_box_note = px.box(
            ag, x="Profil", y="Note_Moy", color="Profil",
            title=t("Spécialistes vs Polyvalents – Note moyenne"),
            category_orders={"Profil": [t("Spécialiste (1-2)"), t("Modéré (3-4)"), t("Polyvalent (5+)")]},
            points="all",
        )
        st.plotly_chart(fig_box_note, use_container_width=True, key="div_box_note")

        # ── Stats summary ──
        st.subheader(t("Résumé par profil"))
        # Interprétation dynamique
        st.markdown(interpret_diversity(ag), unsafe_allow_html=True)
        profil_summary = ag.groupby("Profil", observed=True).agg(
            Athlètes=("Nom", "count"),
            Win_Rate_Moy=("Win_Rate", "mean"),
            Note_Moy=("Note_Moy", "mean"),
            Passages_Moy=("Passages", "mean"),
            Tour_Max_Freq=("Tour_Max", lambda s: s.mode().iloc[0] if not s.empty else "—"),
        ).reset_index()
        profil_summary["Win_Rate_Moy"] = profil_summary["Win_Rate_Moy"].round(1)
        profil_summary["Note_Moy"] = profil_summary["Note_Moy"].round(2)
        profil_summary["Passages_Moy"] = profil_summary["Passages_Moy"].round(1)
        st.dataframe(format_display_df(profil_summary), use_container_width=True)

        # ── Top athletes by diversity ──
        st.subheader(t("Classement des athlètes"))

        # Filters for ranking
        rank_col1, rank_col2, rank_col3 = st.columns(3)
        with rank_col1:
            sort_col = st.selectbox(t("Trier par"), ["Win_Rate", "Note_Moy", "Katas_Distincts", "Passages"], format_func=fmt_col, key="div_sort")
        with rank_col2:
            profil_opts = [t("Tous")] + sorted(ag["Profil"].dropna().unique().tolist())
            sel_profil = st.selectbox(t("Profil"), profil_opts, key="div_profil_filter")
        with rank_col3:
            style_opts = [t("Tous")] + sorted(d["Style"].dropna().unique().tolist())
            sel_style_rank = st.selectbox(t("Style"), style_opts, key="div_style_filter")

        ag_filtered = ag.copy()
        if sel_profil != t("Tous"):
            ag_filtered = ag_filtered[ag_filtered["Profil"] == sel_profil]
        if sel_style_rank != t("Tous"):
            # Filter on athletes who primarily use this style
            athletes_style = d[d["Style"] == sel_style_rank]["Nom"].unique()
            ag_filtered = ag_filtered[ag_filtered["Nom"].isin(athletes_style)]

        display_cols = ["Nom", "Profil", "Katas_Distincts", "Passages", "Win_Rate", "Note_Moy", "Note_Max", "Tour_Max"]
        if ag_filtered.empty:
            st.info(t("Aucun athlète avec ces critères."))
        else:
            st.dataframe(
                format_display_df(ag_filtered[display_cols].sort_values(sort_col, ascending=False).reset_index(drop=True).head(30)),
                use_container_width=True,
            )

        # ── Most popular katas ──
        st.subheader(t("Katas les plus utilisés"))
        kata_pop = d.groupby("Kata").agg(
            Utilisations=("Note", "count"),
            Athlètes=("Nom", "nunique"),
            Note_Moy=("Note", "mean"),
            Win_Rate=("Victoire", "mean"),
        ).reset_index()
        kata_pop["Note_Moy"] = kata_pop["Note_Moy"].round(2)
        kata_pop["Win_Rate"] = (kata_pop["Win_Rate"] * 100).round(1)
        kata_pop = kata_pop.sort_values("Utilisations", ascending=False)
        st.dataframe(format_display_df(kata_pop), use_container_width=True)

        fig_kata = px.bar(
            kata_pop.head(20), x="Kata", y="Utilisations",
            color="Win_Rate", color_continuous_scale="RdYlGn",
            hover_data=["Athlètes", "Note_Moy", "Win_Rate"],
            title=t("Top 20 katas – Nombre d'utilisations"),
        )
        fig_kata.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_kata, use_container_width=True, key="div_kata_pop")
