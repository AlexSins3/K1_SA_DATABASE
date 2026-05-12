# tabs/score_differential.py — Onglet B : Marge de victoire / score différentiel
from __future__ import annotations

import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

from utils.ui import filter_panel_open, filter_panel_close
from utils.interpretations import show_tab_help, show_chart_guide, interpret_score_differential, margin_color, _color_badge
from utils.display import fmt_tour, fmt_df

_TOUR_ORDER = [
    "T1", "T2", "T3",
    "Pool_1", "Pool_2", "Pool_3",
    "PW1", "PW2", "PW3",
    "R1", "R2",
    "Bronze", "Final",
]


@st.cache_data(show_spinner=False)
def _build_match_margins(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruit les matchs R/B et calcule la marge de notes (vectorisé)."""
    d = df.copy().reset_index(drop=True)
    for col in ["Note", "Year", "Ranking"]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")

    d = d.sort_values(["Competition", "Year", "Type_Compet", "N_Tour"], kind="mergesort").reset_index(drop=True)

    # Vectorized pairing via shift
    belt = d["Ceinture"].astype(str)
    belt_next = belt.shift(-1)
    comp = d["Competition"].astype(str)
    nt = d["N_Tour"].astype(str)
    yr = d["Year"].astype(str)

    mask = (
        belt.isin(["R", "B"])
        & belt_next.isin(["R", "B"])
        & (belt != belt_next)
        & (comp == comp.shift(-1))
        & (nt == nt.shift(-1))
        & (yr == yr.shift(-1))
    )

    idx = mask[mask].index.values
    if len(idx) == 0:
        return pd.DataFrame()

    r1 = d.loc[idx].reset_index(drop=True)
    r2 = d.loc[idx + 1].reset_index(drop=True)
    is_r1_red = r1["Ceinture"].astype(str).values == "R"

    def _pick(col):
        return np.where(is_r1_red, r1[col].values, r2[col].values)

    def _pick_inv(col):
        return np.where(is_r1_red, r2[col].values, r1[col].values)

    note_r = np.where(is_r1_red, r1["Note"].values, r2["Note"].values).astype(float)
    note_b = np.where(is_r1_red, r2["Note"].values, r1["Note"].values).astype(float)

    # Filter out rows with NaN notes
    valid = ~(np.isnan(note_r) | np.isnan(note_b))

    result = pd.DataFrame({
        "Competition": _pick("Competition")[valid],
        "Year": _pick("Year")[valid],
        "Type_Compet": _pick("Type_Compet")[valid],
        "N_Tour": _pick("N_Tour")[valid],
        "Red": _pick("Nom")[valid],
        "Blue": _pick_inv("Nom")[valid],
        "Note_Red": note_r[valid],
        "Note_Blue": note_b[valid],
        "Sexe": _pick("Sexe")[valid] if "Sexe" in d.columns else "Inconnu",
    })

    result["Marge_Signée"] = result["Note_Red"] - result["Note_Blue"]
    result["Marge"] = result["Marge_Signée"].abs()
    result["Vainqueur"] = np.where(
        result["Marge_Signée"] > 0, result["Red"],
        np.where(result["Marge_Signée"] < 0, result["Blue"], "Égalité"),
    )
    result["Type_Victoire"] = np.where(
        result["Marge"] > 1.5, "Dominante (>1.5)",
        np.where(result["Marge"] >= 0.5, "Nette (0.5-1.5)", "Serrée (<0.5)"),
    )

    return result


@st.fragment
def show_score_differential_tab(data: pd.DataFrame) -> None:
    st.header("Score différentiel – Marge de victoire")
    show_tab_help("score_differential")

    df = data.copy()
    matches = _build_match_margins(df)

    if matches.empty:
        st.warning("Impossible de reconstruire des matchs avec notes.")
        return

    filters_col, content_col = st.columns([0.9, 2.4])

    with filters_col:
        filter_panel_open()

        sexe_opts = sorted(matches["Sexe"].dropna().unique().tolist()) if "Sexe" in matches.columns else []
        sel_sexe_sd = st.selectbox("Sexe", ["Tous"] + sexe_opts, key="sdiff_sexe")

        tours = sorted(matches["N_Tour"].unique().tolist())
        sel_tours = st.multiselect("Tours", tours, default=tours, format_func=fmt_tour, key="sdiff_tours")

        filter_panel_close()

    with content_col:
        m = matches.copy()
        if sel_sexe_sd != "Tous" and "Sexe" in m.columns:
            m = m[m["Sexe"] == sel_sexe_sd]
        if sel_tours:
            m = m[m["N_Tour"].isin(sel_tours)]

        if m.empty:
            st.info("Aucun match dans ce périmètre.")
            return

        # ── KPIs ──
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Matchs", len(m), help="Nombre total de matchs reconstruits dans le périmètre")
        c2.metric("Marge moy.", f"{m['Marge'].mean():.2f}", help="Écart moyen entre gagnant et perdant. Plus c'est bas, plus les niveaux sont proches")
        c3.metric("Marge médiane", f"{m['Marge'].median():.2f}", help="La marge \"du milieu\" : 50% des matchs ont une marge inférieure")
        pct_serres = (m['Marge'] < 0.5).mean()
        c4.metric("% serrés (<0.5)", f"{pct_serres:.0%}", help="Proportion de matchs très disputés (<0.5 pts d'écart). Plus c'est haut, plus la compétition est relevée")

        # ── Histogram of margins ──
        st.subheader("Distribution de la marge de victoire")
        show_chart_guide("histogram")
        st.markdown(interpret_score_differential(m), unsafe_allow_html=True)

        # Build K1/SA subsets
        m_k1 = m[m["Type_Compet"] == "K1"]
        m_sa = m[m["Type_Compet"] == "SA"]
        _type_order = {"Type_Victoire": ["Serrée (<0.5)", "Nette (0.5-1.5)", "Dominante (>1.5)"]}

        col_k1, col_sa = st.columns(2)
        with col_k1:
            st.markdown("**Premier League (K1)**")
            if m_k1.empty:
                st.info("Aucun match K1.")
            else:
                fig_hist_k1 = px.histogram(
                    m_k1, x="Marge", nbins=30, color="Type_Victoire",
                    title="Marge – K1",
                    labels={"Marge": "Marge (|Note_R − Note_B|)"},
                    category_orders=_type_order,
                )
                st.plotly_chart(fig_hist_k1, use_container_width=True, key="sdiff_hist_k1")
        with col_sa:
            st.markdown("**Series A (SA)**")
            if m_sa.empty:
                st.info("Aucun match SA.")
            else:
                fig_hist_sa = px.histogram(
                    m_sa, x="Marge", nbins=30, color="Type_Victoire",
                    title="Marge – SA",
                    labels={"Marge": "Marge (|Note_R − Note_B|)"},
                    category_orders=_type_order,
                )
                st.plotly_chart(fig_hist_sa, use_container_width=True, key="sdiff_hist_sa")

        # ── Boxplot by tour ──
        st.subheader("Marge par tour")
        show_chart_guide("boxplot")
        col_k1b, col_sab = st.columns(2)
        with col_k1b:
            st.markdown("**K1**")
            if not m_k1.empty:
                m_k1_chart = fmt_df(m_k1)
                fig_box_k1 = px.box(
                    m_k1_chart, x="N_Tour", y="Marge", color="N_Tour",
                    title="Marge par tour – K1",
                    category_orders={"N_Tour": [fmt_tour(t) for t in sorted(m_k1["N_Tour"].unique(), key=lambda t: _TOUR_ORDER.index(t) if t in _TOUR_ORDER else 999)]},
                )
                fig_box_k1.update_layout(showlegend=False)
                st.plotly_chart(fig_box_k1, use_container_width=True, key="sdiff_box_k1")
            else:
                st.info("Aucun match K1.")
        with col_sab:
            st.markdown("**SA**")
            if not m_sa.empty:
                m_sa_chart = fmt_df(m_sa)
                fig_box_sa = px.box(
                    m_sa_chart, x="N_Tour", y="Marge", color="N_Tour",
                    title="Marge par tour – SA",
                    category_orders={"N_Tour": [fmt_tour(t) for t in sorted(m_sa["N_Tour"].unique(), key=lambda t: _TOUR_ORDER.index(t) if t in _TOUR_ORDER else 999)]},
                )
                fig_box_sa.update_layout(showlegend=False)
                st.plotly_chart(fig_box_sa, use_container_width=True, key="sdiff_box_sa")
            else:
                st.info("Aucun match SA.")

        # ── Type de victoire par tour ──
        st.subheader("Type de victoire par tour")
        col_k1c, col_sac = st.columns(2)
        with col_k1c:
            st.markdown("**K1**")
            if not m_k1.empty:
                cross_k1 = m_k1.groupby(["N_Tour", "Type_Victoire"]).size().reset_index(name="Count")
                cross_k1_chart = fmt_df(cross_k1)
                fig_bar_k1 = px.bar(
                    cross_k1_chart, x="N_Tour", y="Count", color="Type_Victoire",
                    barmode="group", category_orders=_type_order,
                    title="Type de victoire par tour – K1",
                )
                st.plotly_chart(fig_bar_k1, use_container_width=True, key="sdiff_type_k1")
            else:
                st.info("Aucun match K1.")
        with col_sac:
            st.markdown("**SA**")
            if not m_sa.empty:
                cross_sa = m_sa.groupby(["N_Tour", "Type_Victoire"]).size().reset_index(name="Count")
                cross_sa_chart = fmt_df(cross_sa)
                fig_bar_sa = px.bar(
                    cross_sa_chart, x="N_Tour", y="Count", color="Type_Victoire",
                    barmode="group", category_orders=_type_order,
                    title="Type de victoire par tour – SA",
                )
                st.plotly_chart(fig_bar_sa, use_container_width=True, key="sdiff_type_sa")
            else:
                st.info("Aucun match SA.")

        # ── Scatter: marge par compétition ──
        st.subheader("Marge par compétition")
        col_k1d, col_sad = st.columns(2)
        with col_k1d:
            st.markdown("**K1**")
            if not m_k1.empty:
                comp_k1 = m_k1.groupby("Competition").agg(
                    Marge_Moy=("Marge", "mean"), Marge_Med=("Marge", "median"), Nb_Matchs=("Marge", "count"),
                ).reset_index().sort_values("Marge_Moy", ascending=False)
                fig_ck1 = px.bar(comp_k1, x="Competition", y="Marge_Moy", hover_data=["Marge_Med", "Nb_Matchs"], title="Marge moy. par compétition – K1")
                fig_ck1.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_ck1, use_container_width=True, key="sdiff_comp_k1")
            else:
                st.info("Aucun match K1.")
        with col_sad:
            st.markdown("**SA**")
            if not m_sa.empty:
                comp_sa = m_sa.groupby("Competition").agg(
                    Marge_Moy=("Marge", "mean"), Marge_Med=("Marge", "median"), Nb_Matchs=("Marge", "count"),
                ).reset_index().sort_values("Marge_Moy", ascending=False)
                fig_csa = px.bar(comp_sa, x="Competition", y="Marge_Moy", hover_data=["Marge_Med", "Nb_Matchs"], title="Marge moy. par compétition – SA")
                fig_csa.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_csa, use_container_width=True, key="sdiff_comp_sa")
            else:
                st.info("Aucun match SA.")
