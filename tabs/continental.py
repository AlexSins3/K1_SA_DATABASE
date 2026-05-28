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
    for col in ["Note", "Year", "Drapeau"]:
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

        # Year filter
        years_available = sorted(d["Year"].dropna().unique().tolist())
        sel_years_cont = st.multiselect(
            t("Année(s)"), [int(y) for y in years_available],
            default=[int(y) for y in years_available], key="cont_years"
        )
        if sel_years_cont:
            d = d[d["Year"].isin([float(y) for y in sel_years_cont])]

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

        # Filter out placeholder rows
        d = d[d["Nom"].notna() & (d["Nom"].astype(str).str.strip() != "")]

        # ── Aggregation ──
        grp = d.groupby(geo_col).agg(
            Athlètes=("Nom", "nunique"),
            Passages=("Victoire", "count"),
            Victoires=("Victoire", "sum"),
            Total_Matchs=("Victoire", "count"),
        ).reset_index()

        # Note stats only where available
        d_with_notes = d.dropna(subset=["Note"])
        if not d_with_notes.empty:
            note_grp = d_with_notes.groupby(geo_col).agg(
                Note_Moy=("Note", "mean"),
                Note_Max=("Note", "max"),
            ).reset_index()
            grp = grp.merge(note_grp, on=geo_col, how="left")
        else:
            grp["Note_Moy"] = np.nan
            grp["Note_Max"] = np.nan

        # Drapeau stats (2026+)
        d_with_flags = d[d["Drapeau"].notna()] if "Drapeau" in d.columns else pd.DataFrame()
        if not d_with_flags.empty:
            flag_grp = d_with_flags.groupby(geo_col).agg(
                Drapeau_Moy=("Drapeau", "mean"),
            ).reset_index()
            grp = grp.merge(flag_grp, on=geo_col, how="left")
        else:
            grp["Drapeau_Moy"] = np.nan

        grp["Win_Rate"] = (grp["Victoires"] / grp["Total_Matchs"] * 100).round(1)
        grp["Note_Moy"] = grp["Note_Moy"].round(2)
        grp["Drapeau_Moy"] = grp["Drapeau_Moy"].round(2)
        grp = grp.sort_values("Win_Rate", ascending=False)

        # Seuil de représentativité : au moins 5 athlètes distincts
        _MIN_ATHLETES = 5
        grp_rep = grp[grp["Athlètes"] >= _MIN_ATHLETES].copy()

        # ── KPIs ──
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"{geo_col}s {t('représentés')}", int(grp[geo_col].nunique()), help=f"{t('Nombre de')} {geo_col.lower()}s {t('différents dans la sélection')}")
        c2.metric(t("Athlètes total"), int(d["Nom"].nunique()), help=t("Nombre total d'athlètes distincts"))
        note_mean = d_with_notes['Note'].mean() if not d_with_notes.empty else np.nan
        c3.metric(t("Note moy. globale") + " (< 2026)", f"{note_mean:.2f}" if pd.notna(note_mean) else "—", help=t("Note moyenne de tous les passages dans ce périmètre"))
        win_rate_global = (d["Victoire"].sum() / len(d) * 100) if len(d) > 0 else 0
        c4.metric(t("Win Rate global"), f"{win_rate_global:.1f}%")

        # ── Table ──
        st.subheader(f"{t('Performances par')} {geo_col}")
        st.caption(f"⚠️ {t('Le Top 3 et les graphiques excluent les')} {geo_col.lower()}s {t('avec moins de')} {_MIN_ATHLETES} {t('athlètes (non représentatifs).')}")
        st.markdown(interpret_continental(grp_rep, geo_col), unsafe_allow_html=True)
        st.dataframe(format_display_df(grp.rename(columns={
            "Athlètes": t("Athlètes"), "Passages": t("Passages"),
            "Victoires": t("Victoires"), "Total_Matchs": t("Total matchs"),
            "Note_Moy": t("Note moy."), "Note_Max": t("Note max"),
            "Drapeau_Moy": t("Drapeau moy."), "Win_Rate": t("Win Rate"),
        })), use_container_width=True)

        # ── Bar chart: win rate (representative only) ──
        top20 = grp_rep.head(20)
        if not top20.empty:
            color_col = "Note_Moy" if grp_rep["Note_Moy"].notna().any() else "Win_Rate"
            fig_wr = px.bar(
                top20, x=geo_col, y="Win_Rate",
                color=color_col, color_continuous_scale="RdYlGn",
                hover_data=["Athlètes", "Passages"],
                title=f"Win rate {t('par')} {geo_col} (Top 20, min {_MIN_ATHLETES} {t('athlètes')})",
                labels={"Win_Rate": "Win rate (%)", "Athlètes": t("Athlètes"), "Passages": t("Passages")},
            )
            fig_wr.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_wr, use_container_width=True, key="cont_wr_bar")

        # ── Note distribution by geo (representative only) — avant 2026 ──
        if not d_with_notes.empty:
            st.subheader(f"{t('Distribution des notes par')} {geo_col} ({t('avant 2026')})")
            show_chart_guide("boxplot")
            big_geos = grp_rep[grp_rep["Passages"] >= 10][geo_col].tolist()
            d_big_notes = d_with_notes[d_with_notes[geo_col].isin(big_geos)]
            if not d_big_notes.empty:
                fig_box = px.box(
                    d_big_notes, x=geo_col, y="Note", color=geo_col,
                    title=f"{t('Notes par')} {geo_col} ({t('avant 2026')}, min {_MIN_ATHLETES} {t('athlètes')} & 10 {t('passages')})",
                )
                fig_box.update_layout(showlegend=False, xaxis_tickangle=-45, height=500)
                st.plotly_chart(fig_box, use_container_width=True, key="cont_box")

            # ── Heatmap: note moyenne par geo × tour ──
            if len(big_geos) >= 2:
                st.subheader(f"{t('Note moy.')} {t('par')} {geo_col} × Tour ({t('avant 2026')})")
                show_chart_guide("heatmap")
                pivot = d_big_notes.groupby([geo_col, "N_Tour"])["Note"].mean().reset_index()
                pivot_chart = fmt_df(pivot)
                pivot_wide = pivot_chart.pivot(index=geo_col, columns="N_Tour", values="Note")
                fig_heat = px.imshow(
                    pivot_wide, aspect="auto", color_continuous_scale="RdYlGn",
                    title=f"Heatmap : {t('Note moy.')} {t('par')} {geo_col} × tour ({t('avant 2026')})",
                    labels=dict(color=t("Note moy.")),
                )
                fig_heat.update_layout(height=max(400, len(big_geos) * 25))
                st.plotly_chart(fig_heat, use_container_width=True, key="cont_heat")

        # ── Drapeau analysis (2026+) ──
        if not d_with_flags.empty:
            st.subheader(f"{t('Analyse drapeaux par')} {geo_col} (2026+)")
            big_geos_f = grp_rep[grp_rep["Passages"] >= 10][geo_col].tolist()
            d_big_flags = d_with_flags[d_with_flags[geo_col].isin(big_geos_f)]
            if not d_big_flags.empty:
                flag_geo = d_big_flags.groupby(geo_col).agg(
                    Drapeau_Moy=("Drapeau", "mean"),
                    Win_Rate=("Victoire", "mean"),
                ).reset_index()
                flag_geo["Win_Rate"] = (flag_geo["Win_Rate"] * 100).round(1)
                flag_geo["Drapeau_Moy"] = flag_geo["Drapeau_Moy"].round(2)
                fig_flag = px.bar(
                    flag_geo.sort_values("Win_Rate", ascending=False).head(20),
                    x=geo_col, y="Win_Rate", color="Drapeau_Moy",
                    color_continuous_scale="RdYlGn",
                    title=f"Win rate (2026+) {t('par')} {geo_col}",
                    labels={"Win_Rate": "Win rate (%)", "Drapeau_Moy": t("Drapeau moy.")},
                )
                fig_flag.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_flag, use_container_width=True, key="cont_flag_bar")

        # ── Finalists analysis ──
        st.subheader(t("Finalistes par") + " " + geo_col)
        finals = d[d["N_Tour"].astype(str).isin(["Final", "Bronze"])]
        if not finals.empty:
            fin_grp = finals.groupby(geo_col).agg(
                Finalistes=("Nom", "nunique"),
                Finales=("Victoire", "count"),
                Victoires_Finale=("Victoire", "sum"),
            ).reset_index().sort_values("Finales", ascending=False)
            st.dataframe(format_display_df(fin_grp.rename(columns={
                "Finalistes": t("Finalistes"),
                "Finales": t("Finales"),
                "Victoires_Finale": t("Victoires en finale"),
            })), use_container_width=True)
        else:
            st.info(t("Aucune donnée de finale dans ce périmètre."))
