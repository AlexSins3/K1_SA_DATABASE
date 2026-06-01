# tabs/match_analysis.py — Analyse des matchs (fusion score différentiel + avancement par tour)
from __future__ import annotations

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.ui import filter_panel_open, filter_panel_close
from utils.interpretations import show_tab_help, show_chart_guide, interpret_score_differential, margin_color, _color_badge
from utils.display import fmt_tour, fmt_df, format_display_df
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


# ═══════════════════════════════════════════════════════════════════════════════
# Match pairing (reused from score_differential)
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _build_match_margins(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruit les matchs R/B et calcule la marge."""
    d = df.copy().reset_index(drop=True)
    for col in ["Note", "Year", "Ranking", "Drapeau"]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")

    d = d[d["Nom"].notna() & (d["Nom"].astype(str).str.strip() != "")].reset_index(drop=True)
    d = d.sort_values(["Competition", "Year", "Type_Compet", "N_Tour"], kind="mergesort").reset_index(drop=True)

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

    has_drapeau = "Drapeau" in d.columns
    if has_drapeau:
        flag_r = np.where(is_r1_red, r1["Drapeau"].values, r2["Drapeau"].values).astype(float)
        flag_b = np.where(is_r1_red, r2["Drapeau"].values, r1["Drapeau"].values).astype(float)
    else:
        flag_r = np.full(len(r1), np.nan)
        flag_b = np.full(len(r1), np.nan)

    year_vals = _pick("Year").astype(float)
    type_compet_vals = _pick("Type_Compet")
    is_flag = np.array([is_flag_era(y) for y in year_vals])

    valid_notes = ~(np.isnan(note_r) | np.isnan(note_b))
    valid_flags = ~(np.isnan(flag_r) | np.isnan(flag_b))
    valid = np.where(is_flag, valid_flags, valid_notes)

    result = pd.DataFrame({
        "Competition": _pick("Competition")[valid],
        "Year": year_vals[valid],
        "Type_Compet": type_compet_vals[valid],
        "N_Tour": _pick("N_Tour")[valid],
        "Red": _pick("Nom")[valid],
        "Blue": _pick_inv("Nom")[valid],
        "Red_Kata": _pick("Kata")[valid],
        "Blue_Kata": _pick_inv("Kata")[valid],
        "Note_Red": note_r[valid],
        "Note_Blue": note_b[valid],
        "Flag_Red": flag_r[valid],
        "Flag_Blue": flag_b[valid],
        "Is_Flag_Era": is_flag[valid],
        "Sexe": _pick("Sexe")[valid] if "Sexe" in d.columns else "Inconnu",
    })

    if result.empty:
        return pd.DataFrame()

    result["Marge_Notes"] = (result["Note_Red"] - result["Note_Blue"]).abs()
    result["Marge_Flags"] = (result["Flag_Red"] - result["Flag_Blue"]).abs()
    result["Marge"] = np.where(result["Is_Flag_Era"], result["Marge_Flags"], result["Marge_Notes"])

    result["Vainqueur"] = np.where(
        result["Is_Flag_Era"],
        np.where(result["Flag_Red"] > result["Flag_Blue"], result["Red"],
                 np.where(result["Flag_Red"] < result["Flag_Blue"], result["Blue"], "Draw")),
        np.where(result["Note_Red"] > result["Note_Blue"], result["Red"],
                 np.where(result["Note_Red"] < result["Note_Blue"], result["Blue"], "Draw")),
    )

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


# ═══════════════════════════════════════════════════════════════════════════════
# Main tab
# ═══════════════════════════════════════════════════════════════════════════════

@st.fragment
def show_match_analysis_tab(data: pd.DataFrame) -> None:
    st.header(t("Analyse des matchs"))
    show_tab_help("match_analysis")

    df = data.copy()
    matches = _build_match_margins(df)

    if matches.empty:
        st.warning(t("Impossible de reconstruire des matchs."))
        return

    filters_col, content_col = st.columns([0.9, 2.4])

    with filters_col:
        filter_panel_open()
        st.markdown(t("### 🎛️ Filtres"))

        # Type compet
        type_opts = [t("Tous"), "K1", "SA"]
        sel_type = st.radio(t("Type compétition"), type_opts, key="ma_type")

        # Sexe
        sexe_opts = sorted(matches["Sexe"].dropna().unique().tolist()) if "Sexe" in matches.columns else []
        sel_sexe = st.selectbox(t("Sexe"), [t("Tous")] + sexe_opts, key="ma_sexe")

        # Tours
        tours = sorted(matches["N_Tour"].unique().tolist())
        sel_tours = st.multiselect(t("Tours"), tours, default=tours, format_func=fmt_tour, key="ma_tours")

        # Année
        years_available = sorted(matches["Year"].dropna().unique().tolist())
        sel_years = st.multiselect(t("Année(s)"), [int(y) for y in years_available],
                                   default=[int(y) for y in years_available], key="ma_years")

        st.markdown("---")
        st.markdown(t("#### 🔎 Filtre athlète / kata"))

        # Filtre athlète
        all_athletes = sorted(set(matches["Red"].tolist() + matches["Blue"].tolist()))
        sel_athlete = st.selectbox(
            t("Athlète (optionnel)"), [t("Tous")] + all_athletes, key="ma_athlete"
        )

        # Filtre kata
        all_katas = sorted(set(matches["Red_Kata"].dropna().tolist() + matches["Blue_Kata"].dropna().tolist()))
        sel_kata = st.selectbox(
            t("Kata (optionnel)"), [t("Tous")] + all_katas, key="ma_kata"
        )

        filter_panel_close()

    with content_col:
        # Apply filters
        m = matches.copy()
        if sel_type != t("Tous"):
            m = m[m["Type_Compet"] == sel_type]
        if sel_sexe != t("Tous") and "Sexe" in m.columns:
            m = m[m["Sexe"] == sel_sexe]
        if sel_tours:
            m = m[m["N_Tour"].isin(sel_tours)]
        if sel_years:
            m = m[m["Year"].isin([float(y) for y in sel_years])]
        if sel_athlete != t("Tous"):
            m = m[(m["Red"] == sel_athlete) | (m["Blue"] == sel_athlete)]
        if sel_kata != t("Tous"):
            m = m[(m["Red_Kata"] == sel_kata) | (m["Blue_Kata"] == sel_kata)]

        if m.empty:
            st.info(t("Aucune donnée dans ce périmètre."))
            return

        # ── Sub-tabs for the two views ──
        tab_diff, tab_funnel = st.tabs([
            "⚔️ " + t("Score différentiel"),
            "🏔️ " + t("Avancement par tour"),
        ])

        # ══════════════════════════════════════════════════════════════
        # Sub-tab 1: Score différentiel
        # ══════════════════════════════════════════════════════════════
        with tab_diff:
            _render_score_differential(m, sel_athlete)

        # ══════════════════════════════════════════════════════════════
        # Sub-tab 2: Avancement par tour (funnel)
        # ══════════════════════════════════════════════════════════════
        with tab_funnel:
            _render_tour_funnel(data, sel_type, sel_sexe, sel_years, sel_athlete, sel_kata)


def _render_score_differential(m: pd.DataFrame, sel_athlete: str) -> None:
    """Render score differential analysis."""

    m_notes = m[~m["Is_Flag_Era"]]
    m_flags = m[m["Is_Flag_Era"]]

    # KPIs — séparés par ère
    marge_moy_notes = m_notes["Marge_Notes"].mean() if not m_notes.empty else None
    marge_moy_flags = m_flags["Marge_Flags"].mean() if not m_flags.empty else None
    pct_serres = (m["Type_Victoire"].str.contains("Serr", na=False)).mean() * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(t("Matchs"), len(m))
    if marge_moy_notes is not None and marge_moy_flags is not None:
        col2.metric(t("Marge moy. (notes)"), f"{marge_moy_notes:.2f}")
        col3.metric(t("Marge moy. (drapeaux)"), f"{marge_moy_flags:.1f}")
    elif marge_moy_notes is not None:
        col2.metric(t("Marge moy. (notes)"), f"{marge_moy_notes:.2f}")
        col3.metric(t("% serrés"), f"{pct_serres:.0f}%")
    else:
        col2.metric(t("Marge moy. (drapeaux)"), f"{marge_moy_flags:.1f}" if marge_moy_flags else "—")
        col3.metric(t("% serrés"), f"{pct_serres:.0f}%")
    col4.metric(t("Compétitions"), m["Competition"].nunique())

    # Interprétation
    st.markdown(interpret_score_differential(m), unsafe_allow_html=True)

    # Distribution des marges
    st.subheader(t("Distribution des marges"))
    show_chart_guide("histogram")

    if not m_notes.empty:
        fig_hist = px.histogram(
            m_notes, x="Marge_Notes", nbins=25, marginal="box",
            title=t("Marge de victoire (système notes, avant 2026)"),
            labels={"Marge_Notes": t("Écart de notes")},
            color_discrete_sequence=["#636EFA"],
        )
        fig_hist.add_vline(x=m_notes["Marge_Notes"].mean(), line_dash="dash", line_color="red",
                           annotation_text=t("Moyenne"))
        st.plotly_chart(fig_hist, width="stretch", key="ma_hist_notes")

    if not m_flags.empty:
        # Scores réels par type de compétition
        m_flags_disp = m_flags.copy()
        m_flags_disp["Score_W"] = m_flags_disp[["Flag_Red", "Flag_Blue"]].max(axis=1).astype(int)
        m_flags_disp["Score_L"] = m_flags_disp[["Flag_Red", "Flag_Blue"]].min(axis=1).astype(int)
        m_flags_disp["Score"] = m_flags_disp["Score_W"].astype(str) + "-" + m_flags_disp["Score_L"].astype(str)

        m_flags_sa = m_flags_disp[m_flags_disp["Type_Compet"] == "SA"]
        m_flags_k1 = m_flags_disp[m_flags_disp["Type_Compet"] == "K1"]
        _score_order_sa = ["5-0", "4-1", "3-2"]
        _score_order_k1 = ["7-0", "6-1", "5-2", "4-3"]

        col_sa, col_k1 = st.columns(2)
        with col_sa:
            if not m_flags_sa.empty:
                sc_sa = m_flags_sa["Score"].value_counts().reindex(_score_order_sa, fill_value=0).reset_index()
                sc_sa.columns = ["Score", "Matchs"]
                sc_sa["Pct"] = (sc_sa["Matchs"] / sc_sa["Matchs"].sum() * 100).round(1)
                fig_sa = px.bar(sc_sa, x="Score", y="Matchs", text="Pct",
                    title=t("SA (5 juges)"), color_discrete_sequence=["#00CC96"],
                    category_orders={"Score": _score_order_sa})
                fig_sa.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                st.plotly_chart(fig_sa, width="stretch", key="ma_hist_flags_sa")
            else:
                st.info(t("Aucun match SA."))
        with col_k1:
            if not m_flags_k1.empty:
                sc_k1 = m_flags_k1["Score"].value_counts().reindex(_score_order_k1, fill_value=0).reset_index()
                sc_k1.columns = ["Score", "Matchs"]
                sc_k1["Pct"] = (sc_k1["Matchs"] / sc_k1["Matchs"].sum() * 100).round(1)
                fig_k1 = px.bar(sc_k1, x="Score", y="Matchs", text="Pct",
                    title=t("K1 (7 juges)"), color_discrete_sequence=["#AB63FA"],
                    category_orders={"Score": _score_order_k1})
                fig_k1.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                st.plotly_chart(fig_k1, width="stretch", key="ma_hist_flags_k1")
            else:
                st.info(t("Aucun match K1."))

    # Marge par tour
    st.subheader(t("Marge par tour"))
    show_chart_guide("boxplot")
    m_sorted = m.copy()
    m_sorted["Tour_Rank"] = m_sorted["N_Tour"].apply(_tour_rank)
    m_sorted = m_sorted.sort_values("Tour_Rank")
    m_sorted["Tour_Display"] = m_sorted["N_Tour"].apply(fmt_tour)

    fig_box = px.box(
        m_sorted, x="Tour_Display", y="Marge", points="outliers",
        title=t("Marge de victoire par tour"),
        labels={"Tour_Display": t("Tour"), "Marge": t("Marge")},
    )
    st.plotly_chart(fig_box, width="stretch", key="ma_box_tour")

    # Type de victoire — distingué par ère
    st.subheader(t("Répartition des types de victoire"))

    if not m_notes.empty and not m_flags.empty:
        col_tv_n, col_tv_f = st.columns(2)
        with col_tv_n:
            tv_n = m_notes["Type_Victoire"].value_counts().reset_index()
            tv_n.columns = ["Type", "Nombre"]
            fig_tv_n = px.pie(tv_n, values="Nombre", names="Type",
                              title=t("Système notes (2024-2025)"),
                              color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig_tv_n, width="stretch", key="ma_pie_tv_notes")
        with col_tv_f:
            tv_f = m_flags["Type_Victoire"].value_counts().reset_index()
            tv_f.columns = ["Type", "Nombre"]
            fig_tv_f = px.pie(tv_f, values="Nombre", names="Type",
                              title=t("Système drapeaux (2026+)"),
                              color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_tv_f, width="stretch", key="ma_pie_tv_flags")
    else:
        tv_counts = m["Type_Victoire"].value_counts().reset_index()
        tv_counts.columns = ["Type", "Nombre"]
        fig_pie = px.pie(tv_counts, values="Nombre", names="Type",
                         title=t("Types de victoire"),
                         color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig_pie, width="stretch", key="ma_pie_tv")

    # Légende types de victoire
    with st.expander(t("📖 Légende des types de victoire")):
        if get_lang() == "en":
            st.markdown(
                f"**{t('Système notes (2024-2025)')}**:\n"
                f"- *Close (<0.5)*: score gap less than 0.5 point\n"
                f"- *Clear (0.5-1.5)*: gap between 0.5 and 1.5 points\n"
                f"- *Dominant (>1.5)*: gap greater than 1.5 points\n\n"
                f"**{t('Système drapeaux (2026+)')}**:\n"
                f"- *Close (1 flag)*: SA 3-2 / K1 4-3\n"
                f"- *Clear (2-3 flags)*: SA 4-1 / K1 5-2 or 6-1\n"
                f"- *Unanimous*: SA 5-0 / K1 7-0"
            )
        else:
            st.markdown(
                f"**{t('Système notes (2024-2025)')}** :\n"
                f"- *Serrée (<0.5)* : écart de notes inférieur à 0.5 point\n"
                f"- *Nette (0.5-1.5)* : écart entre 0.5 et 1.5 points\n"
                f"- *Dominante (>1.5)* : écart supérieur à 1.5 points\n\n"
                f"**{t('Système drapeaux (2026+)')}** :\n"
                f"- *Serré (1 drapeau)* : SA 3-2 / K1 4-3\n"
                f"- *Net (2-3 drapeaux)* : SA 4-1 / K1 5-2 ou 6-1\n"
                f"- *Unanime* : SA 5-0 / K1 7-0"
            )

    # Si un athlète est sélectionné, montrer ses marges
    if sel_athlete != t("Tous"):
        st.markdown("---")
        st.subheader(f"📊 {t('Profil de matchs de')} {sel_athlete}")

        m_ath = m[(m["Red"] == sel_athlete) | (m["Blue"] == sel_athlete)].copy()
        m_ath["Is_Winner"] = m_ath["Vainqueur"] == sel_athlete
        wins = m_ath["Is_Winner"].sum()
        total = len(m_ath)
        wr = (wins / total * 100) if total > 0 else 0

        c1, c2, c3 = st.columns(3)
        c1.metric(t("Matchs"), total)
        c2.metric(t("Victoires"), f"{wins} ({wr:.0f}%)")
        c3.metric(t("Marge moy."), f"{m_ath['Marge'].mean():.2f}")

        # Wins vs losses margin
        m_ath_wins = m_ath[m_ath["Is_Winner"]]
        m_ath_losses = m_ath[~m_ath["Is_Winner"]]

        if not m_ath_wins.empty and not m_ath_losses.empty:
            if get_lang() == "en":
                st.markdown(
                    f"- {t('Quand il/elle **gagne** : marge moy.')} = {m_ath_wins['Marge'].mean():.2f}\n"
                    f"- {t('Quand il/elle **perd** : marge moy.')} = {m_ath_losses['Marge'].mean():.2f}"
                )
            else:
                st.markdown(
                    f"- Quand il/elle **gagne** : marge moy. = {m_ath_wins['Marge'].mean():.2f}\n"
                    f"- Quand il/elle **perd** : marge moy. = {m_ath_losses['Marge'].mean():.2f}"
                )


def _render_tour_funnel(data: pd.DataFrame, sel_type, sel_sexe, sel_years, sel_athlete, sel_kata) -> None:
    """Render tour advancement funnel."""

    df = data.copy()
    for col in ["Note", "Year", "Drapeau"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Victoire" in df.columns:
        df["Victoire"] = df["Victoire"].astype(str).str.lower().isin(["true", "1", "vrai", "yes"]).astype(int)

    # Apply same filters
    d = df.copy()
    if sel_type != t("Tous"):
        d = d[d["Type_Compet"] == sel_type]
    if sel_sexe != t("Tous") and "Sexe" in d.columns:
        d = d[d["Sexe"] == sel_sexe]
    if sel_years:
        d = d[d["Year"].isin([float(y) for y in sel_years])]
    if sel_athlete != t("Tous"):
        d = d[d["Nom"] == sel_athlete]
    if sel_kata != t("Tous"):
        d = d[d["Kata"] == sel_kata]

    d = d[d["Nom"].notna() & (d["Nom"].astype(str).str.strip() != "")]

    if d.empty:
        st.info(t("Aucune donnée dans ce périmètre."))
        return

    def _build_tour_counts(data_subset):
        tours_in = sorted(data_subset["N_Tour"].dropna().astype(str).unique().tolist(), key=_tour_rank)
        rows = []
        for t_val in tours_in:
            t_data = data_subset[data_subset["N_Tour"].astype(str) == t_val]
            rows.append({
                "Tour": t_val,
                "Nb athlètes": t_data["Nom"].nunique(),
                "Nb passages": len(t_data),
                "Note moy.": round(t_data["Note"].mean(), 2) if t_data["Note"].notna().any() else None,
            })
        return pd.DataFrame(rows)

    show_chart_guide("funnel")

    # Funnel K1 / SA side by side
    sub_k1 = d[d["Type_Compet"] == "K1"] if sel_type == t("Tous") or sel_type == "K1" else pd.DataFrame()
    sub_sa = d[d["Type_Compet"] == "SA"] if sel_type == t("Tous") or sel_type == "SA" else pd.DataFrame()

    col_k1, col_sa = st.columns(2)
    with col_k1:
        st.markdown("**Premier League (K1)**")
        if sub_k1.empty:
            st.info(t("Aucune donnée K1."))
        else:
            tc_k1 = _build_tour_counts(sub_k1)
            if not tc_k1.empty:
                tc_k1["Tour_Display"] = tc_k1["Tour"].apply(fmt_tour)
                fig_f_k1 = go.Figure(go.Funnel(
                    y=tc_k1["Tour_Display"], x=tc_k1["Nb athlètes"],
                    textinfo="value+percent initial",
                    marker=dict(color=px.colors.sequential.Teal[:len(tc_k1)]),
                ))
                fig_f_k1.update_layout(title="K1", height=350)
                st.plotly_chart(fig_f_k1, width="stretch", key="ma_funnel_k1")

    with col_sa:
        st.markdown("**Series A (SA)**")
        if sub_sa.empty:
            st.info(t("Aucune donnée SA."))
        else:
            tc_sa = _build_tour_counts(sub_sa)
            if not tc_sa.empty:
                tc_sa["Tour_Display"] = tc_sa["Tour"].apply(fmt_tour)
                fig_f_sa = go.Figure(go.Funnel(
                    y=tc_sa["Tour_Display"], x=tc_sa["Nb athlètes"],
                    textinfo="value+percent initial",
                    marker=dict(color=px.colors.sequential.Teal[:len(tc_sa)]),
                ))
                fig_f_sa.update_layout(title="SA", height=350)
                st.plotly_chart(fig_f_sa, width="stretch", key="ma_funnel_sa")

    # Note crescendo by tour
    st.markdown("---")
    st.subheader(t("Note moyenne par tour (crescendo de difficulté)"))

    tc_all = _build_tour_counts(d)
    if not tc_all.empty and tc_all["Note moy."].notna().any():
        tc_all["Tour_Display"] = tc_all["Tour"].apply(fmt_tour)
        tc_display = tc_all.dropna(subset=["Note moy."])
        if not tc_display.empty:
            fig_note_tour = go.Figure(go.Scatter(
                x=tc_display["Tour_Display"], y=tc_display["Note moy."],
                mode="lines+markers",
            ))
            fig_note_tour.update_layout(
                title=t("Évolution de la note moyenne par tour"),
                xaxis_title=t("Tour"),
                yaxis_title=t("Note moyenne"),
            )
            st.plotly_chart(fig_note_tour, width="stretch", key="ma_note_tour")


