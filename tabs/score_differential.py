# tabs/score_differential.py — Onglet B : Marge de victoire / score différentiel
from __future__ import annotations

import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

from utils.ui import filter_panel_open, filter_panel_close
from utils.interpretations import show_tab_help, show_chart_guide, interpret_score_differential, margin_color, _color_badge
from utils.display import fmt_tour, fmt_df
from utils.data_helpers import is_flag_era, classify_match_flags, classify_match_notes
from utils.lang import t, get_lang

_TOUR_ORDER = [
    "T1", "T2", "T3",
    "Pool_1", "Pool_2", "Pool_3",
    "PW1", "PW2", "PW3",
    "RP1", "RP2", "RP3", "RP4",
    "R1", "R2",
    "Bronze", "Final",
]


def _interpret_margin_by_tour(df_subset: pd.DataFrame) -> str:
    """Generate dynamic interpretation for margin-by-tour data."""
    if df_subset.empty or df_subset["N_Tour"].nunique() < 2:
        return ""
    tour_marge = df_subset.groupby("N_Tour")["Marge"].mean()
    early_tours = [t for t in ["T1", "Pool_1", "Pool_2", "Pool_3"] if t in tour_marge.index]
    late_tours = [t for t in ["R1", "R2", "Bronze", "Final"] if t in tour_marge.index]
    if not early_tours or not late_tours:
        return ""
    early_avg = tour_marge[early_tours].mean()
    late_avg = tour_marge[late_tours].mean()
    if early_avg > late_avg + 0.2:
        return t("📊 Les marges se resserrent en phases finales → les matchs décisifs sont plus disputés.")
    elif late_avg > early_avg + 0.2:
        return t("📊 Les marges augmentent en phases finales → les meilleurs creusent l'écart.")
    return t("📊 Les marges restent stables quel que soit le tour → compétitivité constante.")


@st.cache_data(show_spinner=False)
def _build_match_margins(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruit les matchs R/B et calcule la marge (notes OU drapeaux selon l'ère)."""
    d = df.copy().reset_index(drop=True)
    for col in ["Note", "Year", "Ranking", "Drapeau"]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")

    # Filter out placeholder rows (no Nom)
    d = d[d["Nom"].notna() & (d["Nom"].astype(str).str.strip() != "")].reset_index(drop=True)

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

    # Flag values (for 2026+)
    has_drapeau = "Drapeau" in d.columns
    if has_drapeau:
        flag_r = np.where(is_r1_red, r1["Drapeau"].values, r2["Drapeau"].values).astype(float)
        flag_b = np.where(is_r1_red, r2["Drapeau"].values, r1["Drapeau"].values).astype(float)
    else:
        flag_r = np.full(len(r1), np.nan)
        flag_b = np.full(len(r1), np.nan)

    year_vals = _pick("Year").astype(float)
    type_compet_vals = _pick("Type_Compet")

    # Determine which system applies per match
    is_flag = np.array([is_flag_era(y) for y in year_vals])

    # For note era: valid if both notes exist
    valid_notes = ~(np.isnan(note_r) | np.isnan(note_b))
    # For flag era: valid if both flags exist
    valid_flags = ~(np.isnan(flag_r) | np.isnan(flag_b))
    # Overall valid
    valid = np.where(is_flag, valid_flags, valid_notes)

    result = pd.DataFrame({
        "Competition": _pick("Competition")[valid],
        "Year": year_vals[valid],
        "Type_Compet": type_compet_vals[valid],
        "N_Tour": _pick("N_Tour")[valid],
        "Red": _pick("Nom")[valid],
        "Blue": _pick_inv("Nom")[valid],
        "Note_Red": note_r[valid],
        "Note_Blue": note_b[valid],
        "Flag_Red": flag_r[valid],
        "Flag_Blue": flag_b[valid],
        "Is_Flag_Era": is_flag[valid],
        "Sexe": _pick("Sexe")[valid] if "Sexe" in d.columns else "Inconnu",
    })

    if result.empty:
        return pd.DataFrame()

    # Compute margins differently for note era vs flag era
    result["Marge_Notes"] = (result["Note_Red"] - result["Note_Blue"]).abs()
    result["Marge_Flags"] = (result["Flag_Red"] - result["Flag_Blue"]).abs()

    # Unified margin column
    result["Marge"] = np.where(
        result["Is_Flag_Era"],
        result["Marge_Flags"],
        result["Marge_Notes"],
    )

    # Winner
    result["Vainqueur"] = np.where(
        result["Is_Flag_Era"],
        np.where(result["Flag_Red"] > result["Flag_Blue"], result["Red"],
                 np.where(result["Flag_Red"] < result["Flag_Blue"], result["Blue"], "Draw")),
        np.where(result["Note_Red"] > result["Note_Blue"], result["Red"],
                 np.where(result["Note_Red"] < result["Note_Blue"], result["Blue"], "Draw")),
    )

    # Type de victoire — different classification by era
    type_victoire = []
    for i in range(len(result)):
        row = result.iloc[i]
        if row["Is_Flag_Era"]:
            fw = int(max(row["Flag_Red"], row["Flag_Blue"]))
            fl = int(min(row["Flag_Red"], row["Flag_Blue"]))
            type_victoire.append(classify_match_flags(fw, fl, str(row["Type_Compet"])))
        else:
            type_victoire.append(classify_match_notes(row["Marge_Notes"]))
    result["Type_Victoire"] = type_victoire

    return result


@st.fragment
def show_score_differential_tab(data: pd.DataFrame) -> None:
    st.header(t("Score différentiel – Marge de victoire"))
    show_tab_help("score_differential")

    df = data.copy()
    matches = _build_match_margins(df)

    if matches.empty:
        st.warning(t("Impossible de reconstruire des matchs."))
        return

    filters_col, content_col = st.columns([0.9, 2.4])

    with filters_col:
        filter_panel_open()

        sexe_opts = sorted(matches["Sexe"].dropna().unique().tolist()) if "Sexe" in matches.columns else []
        sel_sexe_sd = st.selectbox(t("Sexe"), [t("Tous")] + sexe_opts, key="sdiff_sexe")

        tours = sorted(matches["N_Tour"].unique().tolist())
        sel_tours = st.multiselect(t("Tours"), tours, default=tours, format_func=fmt_tour, key="sdiff_tours")

        # Year / era filter
        years_available = sorted(matches["Year"].dropna().unique().tolist())
        sel_years = st.multiselect(t("Année(s)"), [int(y) for y in years_available], default=[int(y) for y in years_available], key="sdiff_years")

        filter_panel_close()

    with content_col:
        m = matches.copy()
        if sel_sexe_sd != t("Tous") and "Sexe" in m.columns:
            m = m[m["Sexe"] == sel_sexe_sd]
        if sel_tours:
            m = m[m["N_Tour"].isin(sel_tours)]
        if sel_years:
            m = m[m["Year"].isin([float(y) for y in sel_years])]

        if m.empty:
            st.info(t("Aucune donnée dans ce périmètre."))
            return

        # Translate Type_Victoire at display time
        _tv_map_notes = {
            "Serrée (<0.5)": t("Serrée (<0.5)"),
            "Nette (0.5-1.5)": t("Nette (0.5-1.5)"),
            "Dominante (>1.5)": t("Dominante (>1.5)"),
        }
        _tv_map_flags = {
            "Serré (1 drapeau)": t("Serré (1 drapeau)"),
            "Classique": t("Classique"),
            "Déséquilibré": t("Déséquilibré"),
        }
        _tv_map = {**_tv_map_notes, **_tv_map_flags}
        m["Type_Victoire_Display"] = m["Type_Victoire"].map(_tv_map).fillna(m["Type_Victoire"])

        # ── Separate by era for display ──
        m_notes = m[~m["Is_Flag_Era"]]
        m_flags = m[m["Is_Flag_Era"]]

        # ── KPIs ──
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(t("Matchs"), len(m), help=t("Nombre total de matchs reconstruits"))

        if not m_notes.empty:
            c2.metric(t("Marge moy. (notes)"), f"{m_notes['Marge'].mean():.2f}", help=t("Écart moyen notes (2024-2025)"))
        if not m_flags.empty:
            c3.metric(t("Marge moy. (drapeaux)"), f"{m_flags['Marge'].mean():.1f}", help=t("Écart moyen drapeaux (2026+)"))

        if not m_flags.empty:
            pct_serres_flags = (m_flags["Type_Victoire"] == "Serré (1 drapeau)").mean()
            c4.metric(t("% serrés (drapeaux)"), f"{pct_serres_flags:.0%}", help=t("Matchs à 1 drapeau d'écart"))
        elif not m_notes.empty:
            pct_serres = (m_notes['Marge'] < 0.5).mean()
            c4.metric(t("% serrés (<0.5)"), f"{pct_serres:.0%}", help=t("Proportion de matchs très disputés"))

        # ══════ Section Notes (2024-2025) ══════
        if not m_notes.empty:
            st.markdown("---")
            st.subheader(t("Système de notes (2024-2025)"))
            st.markdown(interpret_score_differential(m_notes), unsafe_allow_html=True)

            m_k1_n = m_notes[m_notes["Type_Compet"] == "K1"]
            m_sa_n = m_notes[m_notes["Type_Compet"] == "SA"]
            _type_order_notes = {"Type_Victoire_Display": [t("Serrée (<0.5)"), t("Nette (0.5-1.5)"), t("Dominante (>1.5)")]}

            col_k1, col_sa = st.columns(2)
            with col_k1:
                st.markdown("**Premier League (K1)**")
                if m_k1_n.empty:
                    st.info(t("Aucun match K1."))
                else:
                    fig_hist_k1 = px.histogram(
                        m_k1_n, x="Marge", nbins=30, color="Type_Victoire_Display",
                        title=t("Marge – K1"),
                        labels={"Marge": t("Marge (|Note_R − Note_B|)"), "Type_Victoire_Display": t("Type_Victoire")},
                        category_orders=_type_order_notes,
                    )
                    st.plotly_chart(fig_hist_k1, use_container_width=True, key="sdiff_hist_k1_notes")
            with col_sa:
                st.markdown("**Series A (SA)**")
                if m_sa_n.empty:
                    st.info(t("Aucun match SA."))
                else:
                    fig_hist_sa = px.histogram(
                        m_sa_n, x="Marge", nbins=30, color="Type_Victoire_Display",
                        title=t("Marge – SA"),
                        labels={"Marge": t("Marge (|Note_R − Note_B|)"), "Type_Victoire_Display": t("Type_Victoire")},
                        category_orders=_type_order_notes,
                    )
                    st.plotly_chart(fig_hist_sa, use_container_width=True, key="sdiff_hist_sa_notes")

        # ══════ Section Drapeaux (2026+) ══════
        if not m_flags.empty:
            st.markdown("---")
            st.subheader(t("Système de drapeaux (2026+)"))
            st.caption(t("SA : 5 juges | K1 : 7 juges"))

            # Compute match score label (e.g. "5-0", "4-1", "3-2")
            m_flags = m_flags.copy()
            m_flags["Score_W"] = m_flags[["Flag_Red", "Flag_Blue"]].max(axis=1).astype(int)
            m_flags["Score_L"] = m_flags[["Flag_Red", "Flag_Blue"]].min(axis=1).astype(int)
            m_flags["Score_Label"] = m_flags["Score_W"].astype(str) + "-" + m_flags["Score_L"].astype(str)

            m_k1_f = m_flags[m_flags["Type_Compet"] == "K1"]
            m_sa_f = m_flags[m_flags["Type_Compet"] == "SA"]

            # Ordres possibles de scores
            _score_order_sa = ["5-0", "4-1", "3-2"]
            _score_order_k1 = ["7-0", "6-1", "5-2", "4-3"]

            col_k1f, col_saf = st.columns(2)
            with col_k1f:
                st.markdown(f"**Premier League (K1) — {t('7 juges')}**")
                if m_k1_f.empty:
                    st.info(t("Aucun match K1."))
                else:
                    score_k1 = m_k1_f["Score_Label"].value_counts().reindex(_score_order_k1, fill_value=0).reset_index()
                    score_k1.columns = ["Score", "Matchs"]
                    score_k1["Pct"] = (score_k1["Matchs"] / score_k1["Matchs"].sum() * 100).round(1)
                    fig_k1f = px.bar(
                        score_k1, x="Score", y="Matchs", text="Pct",
                        title=t("Répartition des scores – K1 (7 juges)"),
                        color="Score", color_discrete_sequence=px.colors.sequential.Teal,
                        category_orders={"Score": _score_order_k1},
                    )
                    fig_k1f.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                    fig_k1f.update_layout(showlegend=False)
                    st.plotly_chart(fig_k1f, width="stretch", key="sdiff_bar_k1_flags")

            with col_saf:
                st.markdown(f"**Series A (SA) — {t('5 juges')}**")
                if m_sa_f.empty:
                    st.info(t("Aucun match SA."))
                else:
                    score_sa = m_sa_f["Score_Label"].value_counts().reindex(_score_order_sa, fill_value=0).reset_index()
                    score_sa.columns = ["Score", "Matchs"]
                    score_sa["Pct"] = (score_sa["Matchs"] / score_sa["Matchs"].sum() * 100).round(1)
                    fig_saf = px.bar(
                        score_sa, x="Score", y="Matchs", text="Pct",
                        title=t("Répartition des scores – SA (5 juges)"),
                        color="Score", color_discrete_sequence=px.colors.sequential.Teal,
                        category_orders={"Score": _score_order_sa},
                    )
                    fig_saf.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                    fig_saf.update_layout(showlegend=False)
                    st.plotly_chart(fig_saf, width="stretch", key="sdiff_bar_sa_flags")

        # ══════ Detailed analysis: sub-tabs Notes / Drapeaux ══════
        st.markdown("---")
        tab_notes, tab_flags = st.tabs([t("Analyse par Notes (avant 2026)"), t("Analyse par Drapeaux (2026+)")])

        # ── Sub-tab: Notes ──
        with tab_notes:
            if m_notes.empty:
                st.info(t("Aucune donnée avec notes dans ce périmètre."))
            else:
                # Year sub-tabs for notes
                note_years = sorted(m_notes["Year"].dropna().unique().tolist())
                year_labels_n = [t("Toutes")] + [str(int(y)) for y in note_years]
                year_tabs_n = st.tabs(year_labels_n)

                for yt_idx, yt in enumerate(year_tabs_n):
                    with yt:
                        mn = m_notes if yt_idx == 0 else m_notes[m_notes["Year"] == note_years[yt_idx - 1]]
                        if mn.empty:
                            st.info(t("Aucune donnée pour cette année."))
                            continue

                        mn_k1 = mn[mn["Type_Compet"] == "K1"]
                        mn_sa = mn[mn["Type_Compet"] == "SA"]
                        _key_suffix = f"_n_{yt_idx}"

                        # Marge par tour
                        st.markdown(f"##### {t('Marge par tour')}")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**K1**")
                            if not mn_k1.empty:
                                fig = px.box(fmt_df(mn_k1), x="N_Tour", y="Marge", color="N_Tour",
                                    title=t("Marge par tour – K1"),
                                    category_orders={"N_Tour": [fmt_tour(t_val) for t_val in sorted(mn_k1["N_Tour"].unique(), key=lambda x: _TOUR_ORDER.index(x) if x in _TOUR_ORDER else 999)]})
                                fig.update_layout(showlegend=False)
                                st.plotly_chart(fig, use_container_width=True, key=f"sdiff_box_k1{_key_suffix}")
                        with col2:
                            st.markdown("**SA**")
                            if not mn_sa.empty:
                                fig = px.box(fmt_df(mn_sa), x="N_Tour", y="Marge", color="N_Tour",
                                    title=t("Marge par tour – SA"),
                                    category_orders={"N_Tour": [fmt_tour(t_val) for t_val in sorted(mn_sa["N_Tour"].unique(), key=lambda x: _TOUR_ORDER.index(x) if x in _TOUR_ORDER else 999)]})
                                fig.update_layout(showlegend=False)
                                st.plotly_chart(fig, use_container_width=True, key=f"sdiff_box_sa{_key_suffix}")

                        # Interprétation marge par tour
                        _interp = _interpret_margin_by_tour(mn)
                        if _interp:
                            st.markdown(_interp)

                        # Type de victoire par tour
                        st.markdown(f"##### {t('Type de victoire par tour')}")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**K1**")
                            if not mn_k1.empty:
                                cross = mn_k1.groupby(["N_Tour", "Type_Victoire_Display"]).size().reset_index(name="Count")
                                fig = px.bar(fmt_df(cross), x="N_Tour", y="Count", color="Type_Victoire_Display", barmode="group",
                                    title=t("Type de victoire par tour – K1"))
                                st.plotly_chart(fig, use_container_width=True, key=f"sdiff_type_k1{_key_suffix}")
                        with col2:
                            st.markdown("**SA**")
                            if not mn_sa.empty:
                                cross = mn_sa.groupby(["N_Tour", "Type_Victoire_Display"]).size().reset_index(name="Count")
                                fig = px.bar(fmt_df(cross), x="N_Tour", y="Count", color="Type_Victoire_Display", barmode="group",
                                    title=t("Type de victoire par tour – SA"))
                                st.plotly_chart(fig, use_container_width=True, key=f"sdiff_type_sa{_key_suffix}")

                        # Marge par compétition
                        st.markdown(f"##### {t('Marge par compétition')}")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**K1**")
                            if not mn_k1.empty:
                                comp = mn_k1.groupby("Competition").agg(Marge_Moy=("Marge", "mean"), Nb_Matchs=("Marge", "count")).reset_index().sort_values("Marge_Moy", ascending=False)
                                fig = px.bar(comp, x="Competition", y="Marge_Moy", hover_data=["Nb_Matchs"], title=t("Marge moy. par compétition – K1"),
                                    labels={"Marge_Moy": t("Marge moy."), "Nb_Matchs": t("Nb matchs")})
                                fig.update_layout(xaxis_tickangle=-45)
                                st.plotly_chart(fig, use_container_width=True, key=f"sdiff_comp_k1{_key_suffix}")
                        with col2:
                            st.markdown("**SA**")
                            if not mn_sa.empty:
                                comp = mn_sa.groupby("Competition").agg(Marge_Moy=("Marge", "mean"), Nb_Matchs=("Marge", "count")).reset_index().sort_values("Marge_Moy", ascending=False)
                                fig = px.bar(comp, x="Competition", y="Marge_Moy", hover_data=["Nb_Matchs"], title=t("Marge moy. par compétition – SA"),
                                    labels={"Marge_Moy": t("Marge moy."), "Nb_Matchs": t("Nb matchs")})
                                fig.update_layout(xaxis_tickangle=-45)
                                st.plotly_chart(fig, use_container_width=True, key=f"sdiff_comp_sa{_key_suffix}")

                        # Interprétation dynamique globale pour cette année
                        _marge_moy = mn["Marge"].mean()
                        _pct_serres = (mn["Marge"] < 0.5).mean() * 100
                        _n = len(mn)
                        if get_lang() == "en":
                            _interp_comp = (
                                f"📊 **{_n} matches** — average margin {_marge_moy:.2f} pts, "
                                f"**{_pct_serres:.0f}%** close matches (<0.5 pts). "
                            )
                        else:
                            _interp_comp = (
                                f"📊 **{_n} matchs** — marge moyenne {_marge_moy:.2f} pts, "
                                f"**{_pct_serres:.0f}%** de matchs serrés (<0.5 pts). "
                            )
                        if _pct_serres > 50:
                            _interp_comp += t("→ Niveau très homogène, les détails font la différence.")
                        elif _pct_serres > 30:
                            _interp_comp += t("→ Bon équilibre entre matchs serrés et victoires nettes.")
                        else:
                            _interp_comp += t("→ Les écarts sont fréquents, la hiérarchie est marquée.")
                        st.markdown(_interp_comp)

        # ── Sub-tab: Drapeaux ──
        with tab_flags:
            if m_flags.empty:
                st.info(t("Aucune donnée avec drapeaux dans ce périmètre."))
            else:
                flag_years = sorted(m_flags["Year"].dropna().unique().tolist())
                year_labels_f = [t("Toutes")] + [str(int(y)) for y in flag_years]
                year_tabs_f = st.tabs(year_labels_f)

                for yt_idx, yt in enumerate(year_tabs_f):
                    with yt:
                        mf = m_flags if yt_idx == 0 else m_flags[m_flags["Year"] == flag_years[yt_idx - 1]]
                        if mf.empty:
                            st.info(t("Aucune donnée pour cette année."))
                            continue

                        mf_k1 = mf[mf["Type_Compet"] == "K1"]
                        mf_sa = mf[mf["Type_Compet"] == "SA"]
                        _key_suffix = f"_f_{yt_idx}"

                        # Marge par tour (drapeaux)
                        st.markdown(f"##### {t('Marge par tour')} ({t('drapeaux')})")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**K1 ({t('7 juges')})**")
                            if not mf_k1.empty:
                                fig = px.bar(fmt_df(mf_k1.groupby("N_Tour")["Marge"].mean().reset_index()),
                                    x="N_Tour", y="Marge", title=t("Marge moy. par tour – K1"),
                                    labels={"Marge": t("Marge moy. (drapeaux)")})
                                st.plotly_chart(fig, use_container_width=True, key=f"sdiff_fbox_k1{_key_suffix}")
                        with col2:
                            st.markdown(f"**SA ({t('5 juges')})**")
                            if not mf_sa.empty:
                                fig = px.bar(fmt_df(mf_sa.groupby("N_Tour")["Marge"].mean().reset_index()),
                                    x="N_Tour", y="Marge", title=t("Marge moy. par tour – SA"),
                                    labels={"Marge": t("Marge moy. (drapeaux)")})
                                st.plotly_chart(fig, use_container_width=True, key=f"sdiff_fbox_sa{_key_suffix}")

                        # Type de victoire par tour (drapeaux)
                        st.markdown(f"##### {t('Type de victoire par tour')} ({t('drapeaux')})")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**K1**")
                            if not mf_k1.empty:
                                cross = mf_k1.groupby(["N_Tour", "Type_Victoire_Display"]).size().reset_index(name="Count")
                                fig = px.bar(fmt_df(cross), x="N_Tour", y="Count", color="Type_Victoire_Display", barmode="group",
                                    title=t("Type de victoire par tour – K1"))
                                st.plotly_chart(fig, use_container_width=True, key=f"sdiff_ftype_k1{_key_suffix}")
                        with col2:
                            st.markdown("**SA**")
                            if not mf_sa.empty:
                                cross = mf_sa.groupby(["N_Tour", "Type_Victoire_Display"]).size().reset_index(name="Count")
                                fig = px.bar(fmt_df(cross), x="N_Tour", y="Count", color="Type_Victoire_Display", barmode="group",
                                    title=t("Type de victoire par tour – SA"))
                                st.plotly_chart(fig, use_container_width=True, key=f"sdiff_ftype_sa{_key_suffix}")

                        # Marge par compétition (drapeaux)
                        st.markdown(f"##### {t('Marge par compétition')} ({t('drapeaux')})")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**K1**")
                            if not mf_k1.empty:
                                comp = mf_k1.groupby("Competition").agg(Marge_Moy=("Marge", "mean"), Nb_Matchs=("Marge", "count")).reset_index().sort_values("Marge_Moy", ascending=False)
                                fig = px.bar(comp, x="Competition", y="Marge_Moy", hover_data=["Nb_Matchs"], title=t("Marge moy. par compétition – K1"),
                                    labels={"Marge_Moy": t("Marge moy."), "Nb_Matchs": t("Nb matchs")})
                                fig.update_layout(xaxis_tickangle=-45)
                                st.plotly_chart(fig, use_container_width=True, key=f"sdiff_fcomp_k1{_key_suffix}")
                        with col2:
                            st.markdown("**SA**")
                            if not mf_sa.empty:
                                comp = mf_sa.groupby("Competition").agg(Marge_Moy=("Marge", "mean"), Nb_Matchs=("Marge", "count")).reset_index().sort_values("Marge_Moy", ascending=False)
                                fig = px.bar(comp, x="Competition", y="Marge_Moy", hover_data=["Nb_Matchs"], title=t("Marge moy. par compétition – SA"),
                                    labels={"Marge_Moy": t("Marge moy."), "Nb_Matchs": t("Nb matchs")})
                                fig.update_layout(xaxis_tickangle=-45)
                                st.plotly_chart(fig, use_container_width=True, key=f"sdiff_fcomp_sa{_key_suffix}")

                        # Interprétation drapeaux
                        _interp_f = _interpret_margin_by_tour(mf)
                        if _interp_f:
                            st.markdown(_interp_f)
                        # Score distribution summary
                        if "Score_Label" in mf.columns:
                            _top_score = mf["Score_Label"].value_counts().idxmax()
                            _top_pct = mf["Score_Label"].value_counts(normalize=True).iloc[0] * 100
                            if get_lang() == "en":
                                st.markdown(
                                    f"📊 Most frequent score: **{_top_score}** ({_top_pct:.0f}% of matches). "
                                    f"Average margin of {mf['Marge'].mean():.1f} flags."
                                )
                            else:
                                st.markdown(
                                    f"📊 Score le plus fréquent : **{_top_score}** ({_top_pct:.0f}% des matchs). "
                                    f"Marge moyenne de {mf['Marge'].mean():.1f} drapeaux d'écart."
                                )
