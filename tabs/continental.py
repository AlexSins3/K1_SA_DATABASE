# tabs/continental.py — Onglet D : Analyse continentale / nationale
from __future__ import annotations

import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

from utils.ui import filter_panel_open, filter_panel_close
from utils.interpretations import show_tab_help, show_chart_guide, interpret_continental
from utils.display import format_display_df, fmt_df
from utils.lang import t


@st.fragment
def show_continental_tab(data: pd.DataFrame) -> None:
    st.header(t("Analyse continentale & nationale"))
    show_tab_help("continental")

    df = data.copy()
    for col in ["Note", "Year"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Victoire" in df.columns:
        df["Victoire"] = df["Victoire"].astype(str).str.lower().isin(["true", "1", "vrai", "yes"]).astype(int)

    # Check available geo columns
    has_continent = "Continent" in df.columns
    has_region = "Region_monde" in df.columns

    filters_col, content_col = st.columns([0.9, 2.4])

    with filters_col:
        filter_panel_open()

        type_opts = [t("Tous"), "K1", "SA"]
        sel_type = st.radio(t("Type compétition"), type_opts, key="cont_type")
        d = df.copy()
        if sel_type != t("Tous"):
            d = d[d["Type_Compet"] == sel_type]

        sexes = sorted(d["Sexe"].dropna().unique().tolist())
        sel_sexe = st.selectbox(t("Sexe"), [t("Tous")] + sexes, key="cont_sexe")
        if sel_sexe != t("Tous"):
            d = d[d["Sexe"] == sel_sexe]

        # Level of analysis
        geo_col = "Nation"
        if has_continent or has_region:
            options = ["Nation"]
            if has_continent:
                options.append("Continent")
            if has_region:
                options.append("Region_monde")
            geo_col = st.radio(t("Niveau d'analyse"), options, key="cont_level")

        # Nation/geo filter
        all_geos = sorted(d[geo_col].dropna().unique().tolist())
        sel_geos = st.multiselect(f"{t('Filtrer par')} {geo_col}", all_geos, default=[], key="cont_geo_filter")
        if sel_geos:
            d = d[d[geo_col].isin(sel_geos)]

        filter_panel_close()

    with content_col:
        if d.empty:
            st.info(t("Aucune donnée dans ce périmètre."))
            return

        # ── Aggregation ──
        grp = d.groupby(geo_col).agg(
            Athlètes=("Nom", "nunique"),
            Passages=("Note", "count"),
            Note_Moy=("Note", "mean"),
            Note_Max=("Note", "max"),
            Victoires=("Victoire", "sum"),
            Total_Matchs=("Victoire", "count"),
        ).reset_index()
        grp["Win_Rate"] = (grp["Victoires"] / grp["Total_Matchs"] * 100).round(1)
        grp["Note_Moy"] = grp["Note_Moy"].round(2)
        grp = grp.sort_values("Win_Rate", ascending=False)

        # Seuil de représentativité : au moins 5 athlètes distincts
        _MIN_ATHLETES = 5
        grp_rep = grp[grp["Athlètes"] >= _MIN_ATHLETES].copy()

        # ── KPIs ──
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{geo_col}s {t('représentés')}", int(grp[geo_col].nunique()), help=f"{t('Nombre de')} {geo_col.lower()}s {t('différents dans la sélection')}")
        c2.metric(t("Athlètes total"), int(d["Nom"].nunique()), help=t("Nombre total d'athlètes distincts"))
        c3.metric(t("Note moy. globale"), f"{d['Note'].mean():.2f}", help=t("Note moyenne de tous les passages dans ce périmètre"))

        # ── Table ──
        st.subheader(f"{t('Performances par')} {geo_col}")
        st.caption(f"⚠️ {t('Le Top 3 et les graphiques excluent les')} {geo_col.lower()}s {t('avec moins de')} {_MIN_ATHLETES} {t('athlètes (non représentatifs).')}")
        # Interprétation dynamique (sur données représentatives)
        st.markdown(interpret_continental(grp_rep, geo_col), unsafe_allow_html=True)
        st.dataframe(format_display_df(grp), use_container_width=True)

        # ── Bar chart: win rate (representative only) ──
        top20 = grp_rep.head(20)
        if not top20.empty:
            fig_wr = px.bar(
                top20, x=geo_col, y="Win_Rate",
                color="Note_Moy", color_continuous_scale="RdYlGn",
                hover_data=["Athlètes", "Passages", "Note_Moy"],
                title=f"Win rate {t('par')} {geo_col} (Top 20, min {_MIN_ATHLETES} {t('athlètes')})",
                labels={"Win_Rate": "Win rate (%)"},
            )
            fig_wr.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_wr, use_container_width=True, key="cont_wr_bar")

        # ── Note distribution by geo (representative only) ──
        st.subheader(f"{t('Distribution des notes par')} {geo_col}")
        show_chart_guide("boxplot")
        big_geos = grp_rep[grp_rep["Passages"] >= 10][geo_col].tolist()
        d_big = d[d[geo_col].isin(big_geos)]
        if not d_big.empty:
            fig_box = px.box(
                d_big, x=geo_col, y="Note", color=geo_col,
                title=f"{t('Notes par')} {geo_col} (min {_MIN_ATHLETES} {t('athlètes')} & 10 passages)",
            )
            fig_box.update_layout(showlegend=False, xaxis_tickangle=-45, height=500)
            st.plotly_chart(fig_box, use_container_width=True, key="cont_box")

        # ── Heatmap: note moyenne par geo × tour ──
        if len(big_geos) >= 2:
            st.subheader(f"{t('Note moy.')} {t('par')} {geo_col} × Tour")
            show_chart_guide("heatmap")
            pivot = d_big.groupby([geo_col, "N_Tour"])["Note"].mean().reset_index()
            pivot_chart = fmt_df(pivot)
            pivot_wide = pivot_chart.pivot(index=geo_col, columns="N_Tour", values="Note")
            fig_heat = px.imshow(
                pivot_wide, aspect="auto", color_continuous_scale="RdYlGn",
                title=f"Heatmap : {t('Note moy.')} {t('par')} {geo_col} × tour",
                labels=dict(color=t("Note moy.")),
            )
            fig_heat.update_layout(height=max(400, len(big_geos) * 25))
            st.plotly_chart(fig_heat, use_container_width=True, key="cont_heat")

        # ── Finalists analysis ──
        st.subheader(t("Finalistes par") + " " + geo_col)
        finals = d[d["N_Tour"].astype(str).isin(["Final", "Bronze"])]
        if not finals.empty:
            fin_grp = finals.groupby(geo_col).agg(
                Finalistes=("Nom", "nunique"),
                Finales=("Note", "count"),
                Victoires_Finale=("Victoire", "sum"),
            ).reset_index().sort_values("Finales", ascending=False)
            st.dataframe(format_display_df(fin_grp), use_container_width=True)
        else:
            st.info(t("Aucune donnée de finale dans ce périmètre."))
