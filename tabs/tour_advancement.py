# tabs/tour_advancement.py — Onglet E : Probabilité d'avancement par tour (funnel)
from __future__ import annotations

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.ui import filter_panel_open, filter_panel_close
from utils.data_helpers import safe_mode
from utils.interpretations import show_tab_help, show_chart_guide, interpret_tour_advancement
from utils.display import fmt_tour, format_display_df, fmt_df
from utils.lang import t


# Logical tour ordering (earliest → latest)
_TOUR_ORDER = [
    "T1", "T2", "T3",
    "Pool_1", "Pool_2", "Pool_3",
    "PW1", "PW2", "PW3",
    "RP1", "RP2", "RP3", "RP4",
    "R1", "R2",
    "Bronze", "Final",
]


def _tour_rank(tour) -> int:
    try:
        if tour is None or pd.isna(tour):
            return 999
    except (TypeError, ValueError):
        pass
    try:
        return _TOUR_ORDER.index(str(tour))
    except ValueError:
        return 999


@st.fragment
def show_tour_advancement_tab(data: pd.DataFrame) -> None:
    st.header(t("Avancement par tour – Funnel d'élimination"))
    show_tab_help("tour_advancement")

    df = data.copy()
    for col in ["Note", "Year", "Drapeau"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Victoire" in df.columns:
        df["Victoire"] = df["Victoire"].astype(str).str.lower().isin(["true", "1", "vrai", "yes"]).astype(int)

    filters_col, content_col = st.columns([0.9, 2.4])

    with filters_col:
        filter_panel_open()

        sexes = sorted(df["Sexe"].dropna().unique().tolist())
        sel_sexe = st.selectbox(t("Sexe"), [t("Tous")] + sexes, key="adv_sexe")
        d = df.copy()
        if sel_sexe != t("Tous"):
            d = d[d["Sexe"] == sel_sexe]

        # Year filter
        years_avail = sorted(d["Year"].dropna().unique().tolist())
        sel_years_adv = st.multiselect(
            t("Année(s)"), [int(y) for y in years_avail],
            default=[int(y) for y in years_avail], key="adv_years"
        )
        if sel_years_adv:
            d = d[d["Year"].isin([float(y) for y in sel_years_adv])]

        mode = st.radio(t("Analyse"), [t("Global"), t("Par athlète"), t("Par kata"), t("Par style")], key="adv_mode")

        sel_athlete = None
        sel_kata = None
        sel_style = None
        if mode == t("Par athlète"):
            athletes = sorted(d["Nom"].dropna().unique().tolist())
            sel_athlete = st.selectbox(t("Athlète"), athletes, key="adv_athlete") if athletes else None
        elif mode == t("Par kata"):
            katas = sorted(d["Kata"].dropna().unique().tolist())
            sel_kata = st.selectbox(t("Kata"), katas, key="adv_kata") if katas else None
        elif mode == t("Par style"):
            styles = sorted(d["Style"].dropna().unique().tolist())
            sel_style = st.selectbox(t("Style"), styles, key="adv_style") if styles else None

        filter_panel_close()

    with content_col:
        if d.empty:
            st.info(t("Aucune donnée dans ce périmètre."))
            return

        # Apply secondary filter
        sub = d.copy()
        filter_label = t("Global")
        if mode == t("Par athlète") and sel_athlete:
            sub = sub[sub["Nom"] == sel_athlete]
            filter_label = sel_athlete
        elif mode == t("Par kata") and sel_kata:
            sub = sub[sub["Kata"] == sel_kata]
            filter_label = sel_kata
        elif mode == t("Par style") and sel_style:
            sub = sub[sub["Style"] == sel_style]
            filter_label = sel_style

        # ── Helper to build tour counts ──
        def _build_tour_counts(data_subset):
            tours_in = sorted(data_subset["N_Tour"].dropna().astype(str).unique().tolist(), key=_tour_rank)
            rows = []
            for t in tours_in:
                t_data = data_subset[data_subset["N_Tour"].astype(str) == t]
                rows.append({
                    "Tour": t,
                    "Nb athlètes": t_data["Nom"].nunique(),
                    "Nb passages": len(t_data),
                    "Note moy.": round(t_data["Note"].mean(), 2) if t_data["Note"].notna().any() else None,
                })
            return pd.DataFrame(rows)

        # ── Funnel charts K1/SA side by side ──
        st.subheader(f"{t('Funnel d\'avancement')} – {filter_label}")
        show_chart_guide("funnel")

        sub_k1 = sub[sub["Type_Compet"] == "K1"]
        sub_sa = sub[sub["Type_Compet"] == "SA"]

        col_k1, col_sa = st.columns(2)
        with col_k1:
            st.markdown("**Premier League (K1)**")
            if sub_k1.empty:
                st.info(t("Aucune donnée K1."))
            else:
                tc_k1 = _build_tour_counts(sub_k1)
                tc_k1_disp = tc_k1.copy()
                tc_k1_disp["Tour"] = tc_k1_disp["Tour"].apply(fmt_tour)
                fig_f_k1 = go.Figure(go.Funnel(
                    y=tc_k1_disp["Tour"], x=tc_k1_disp["Nb athlètes"],
                    textinfo="value+percent initial",
                    marker=dict(color=px.colors.sequential.Teal[:len(tc_k1_disp)]),
                ))
                fig_f_k1.update_layout(title="K1", height=400)
                st.plotly_chart(fig_f_k1, use_container_width=True, key="adv_funnel_k1")
                st.markdown(interpret_tour_advancement(tc_k1, f"{filter_label} – K1"), unsafe_allow_html=True)
        with col_sa:
            st.markdown("**Series A (SA)**")
            if sub_sa.empty:
                st.info(t("Aucune donnée SA."))
            else:
                tc_sa = _build_tour_counts(sub_sa)
                tc_sa_disp = tc_sa.copy()
                tc_sa_disp["Tour"] = tc_sa_disp["Tour"].apply(fmt_tour)
                fig_f_sa = go.Figure(go.Funnel(
                    y=tc_sa_disp["Tour"], x=tc_sa_disp["Nb athlètes"],
                    textinfo="value+percent initial",
                    marker=dict(color=px.colors.sequential.Teal[:len(tc_sa_disp)]),
                ))
                fig_f_sa.update_layout(title="SA", height=400)
                st.plotly_chart(fig_f_sa, use_container_width=True, key="adv_funnel_sa")
                st.markdown(interpret_tour_advancement(tc_sa, f"{filter_label} – SA"), unsafe_allow_html=True)

        # ── Performance by tour: sub-tabs Notes / Drapeaux ──
        st.markdown("---")
        tab_adv_notes, tab_adv_flags = st.tabs([t("Notes (avant 2026)"), t("Drapeaux (2026+)")])

        with tab_adv_notes:
            sub_notes = sub[sub["Note"].notna()]
            if sub_notes.empty:
                st.info(t("Aucune donnée avec notes dans ce périmètre."))
            else:
                st.subheader(t("Note moyenne par tour") + f" ({t('avant 2026')})")
                sub_k1_n = sub_notes[sub_notes["Type_Compet"] == "K1"]
                sub_sa_n = sub_notes[sub_notes["Type_Compet"] == "SA"]
                col_k1n, col_san = st.columns(2)
                with col_k1n:
                    st.markdown("**K1**")
                    if not sub_k1_n.empty:
                        tc_k1_notes = _build_tour_counts(sub_k1_n).dropna(subset=["Note moy."])
                        tc_k1_notes["Tour_Display"] = tc_k1_notes["Tour"].apply(fmt_tour)
                        if not tc_k1_notes.empty:
                            fig_n_k1 = px.line(tc_k1_notes, x="Tour_Display", y="Note moy.", markers=True, title=t("Note moy. par tour – K1"))
                            st.plotly_chart(fig_n_k1, use_container_width=True, key="adv_notes_k1")
                with col_san:
                    st.markdown("**SA**")
                    if not sub_sa_n.empty:
                        tc_sa_notes = _build_tour_counts(sub_sa_n).dropna(subset=["Note moy."])
                        tc_sa_notes["Tour_Display"] = tc_sa_notes["Tour"].apply(fmt_tour)
                        if not tc_sa_notes.empty:
                            fig_n_sa = px.line(tc_sa_notes, x="Tour_Display", y="Note moy.", markers=True, title=t("Note moy. par tour – SA"))
                            st.plotly_chart(fig_n_sa, use_container_width=True, key="adv_notes_sa")

        with tab_adv_flags:
            sub_flags = sub[sub["Drapeau"].notna()] if "Drapeau" in sub.columns else pd.DataFrame()
            if sub_flags.empty:
                st.info(t("Aucune donnée avec drapeaux dans ce périmètre."))
            else:
                st.subheader(t("Statistiques drapeaux par tour") + " (2026+)")
                sub_k1_f = sub_flags[sub_flags["Type_Compet"] == "K1"]
                sub_sa_f = sub_flags[sub_flags["Type_Compet"] == "SA"]

                def _build_flag_tour_stats(data_subset):
                    tours_in = sorted(data_subset["N_Tour"].dropna().astype(str).unique().tolist(), key=_tour_rank)
                    rows = []
                    for tour in tours_in:
                        t_data = data_subset[data_subset["N_Tour"].astype(str) == tour]
                        rows.append({
                            "Tour": tour,
                            "Nb athlètes": t_data["Nom"].nunique(),
                            "Drapeau moy.": round(t_data["Drapeau"].mean(), 2),
                            "Win Rate": f"{t_data['Victoire'].mean():.0%}" if t_data["Victoire"].notna().any() else "—",
                        })
                    return pd.DataFrame(rows)

                col_k1f, col_saf = st.columns(2)
                with col_k1f:
                    st.markdown(f"**K1 ({t('7 juges')})**")
                    if not sub_k1_f.empty:
                        tc_k1_f = _build_flag_tour_stats(sub_k1_f)
                        tc_k1_f["Tour_Display"] = tc_k1_f["Tour"].apply(fmt_tour)
                        fig_fk1 = px.bar(tc_k1_f, x="Tour_Display", y="Drapeau moy.", title=t("Drapeau moy. par tour – K1"),
                                         labels={"Drapeau moy.": t("Drapeau moy."), "Tour_Display": "Tour"})
                        st.plotly_chart(fig_fk1, use_container_width=True, key="adv_flags_k1")
                        st.dataframe(format_display_df(tc_k1_f[["Tour", "Nb athlètes", "Drapeau moy.", "Win Rate"]].rename(columns={"Nb athlètes": t("Nb athlètes"), "Drapeau moy.": t("Drapeau moy.")})), use_container_width=True)
                with col_saf:
                    st.markdown(f"**SA ({t('5 juges')})**")
                    if not sub_sa_f.empty:
                        tc_sa_f = _build_flag_tour_stats(sub_sa_f)
                        tc_sa_f["Tour_Display"] = tc_sa_f["Tour"].apply(fmt_tour)
                        fig_fsa = px.bar(tc_sa_f, x="Tour_Display", y="Drapeau moy.", title=t("Drapeau moy. par tour – SA"),
                                         labels={"Drapeau moy.": t("Drapeau moy."), "Tour_Display": "Tour"})
                        st.plotly_chart(fig_fsa, use_container_width=True, key="adv_flags_sa")
                        st.dataframe(format_display_df(tc_sa_f[["Tour", "Nb athlètes", "Drapeau moy.", "Win Rate"]].rename(columns={"Nb athlètes": t("Nb athlètes"), "Drapeau moy.": t("Drapeau moy.")})), use_container_width=True)

        # ── Top athletes reaching advanced tours ──
        if mode == t("Global"):
            st.subheader(t("Top athlètes – Tours avancés atteints"))

            # Filters for top athletes
            top_col1, top_col2, top_col3 = st.columns(3)
            with top_col1:
                top_type = st.selectbox(t("Type compétition"), ["K1", "SA"], key="adv_top_type")
            with top_col2:
                top_sexe_opts = sorted(d["Sexe"].dropna().unique().tolist())
                top_sexe = st.selectbox(t("Sexe (top)"), top_sexe_opts, key="adv_top_sexe") if top_sexe_opts else None
            with top_col3:
                all_tours = sorted(d["N_Tour"].dropna().astype(str).unique().tolist(), key=_tour_rank)
                top_tours = st.multiselect(t("Tours (min atteint)"), all_tours, default=all_tours, format_func=fmt_tour, key="adv_top_tours")

            sub_top = d[d["Type_Compet"] == top_type].copy()
            if top_sexe:
                sub_top = sub_top[sub_top["Sexe"] == top_sexe]
            if top_tours:
                sub_top = sub_top[sub_top["N_Tour"].astype(str).isin(top_tours)]

            if not sub_top.empty:
                athlete_max_tour = sub_top.groupby("Nom")["N_Tour"].apply(
                    lambda s: max(s.astype(str), key=_tour_rank) if len(s) > 0 else None
                ).dropna().reset_index()
                athlete_max_tour.columns = ["Nom", "Tour_Max"]
                athlete_max_tour["Tour_Rank"] = athlete_max_tour["Tour_Max"].apply(_tour_rank)
                athlete_max_tour = athlete_max_tour.sort_values("Tour_Rank", ascending=False).head(20)

                note_means = sub_top.groupby("Nom")["Note"].mean().reset_index()
                note_means.columns = ["Nom", "Note_Moy"]
                athlete_max_tour = athlete_max_tour.merge(note_means, on="Nom", how="left")
                athlete_max_tour["Note_Moy"] = athlete_max_tour["Note_Moy"].round(2)
                st.dataframe(
                    format_display_df(athlete_max_tour[["Nom", "Tour_Max", "Note_Moy"]].reset_index(drop=True)),
                    use_container_width=True,
                )
            else:
                st.info(t("Aucune donnée avec ces filtres."))
