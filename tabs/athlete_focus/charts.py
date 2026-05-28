# tabs/athlete_focus/charts.py

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from constants.tours import (
    K1_TOUR_ORDER, SA_TOUR_ORDER,
    KIVIAT_TOUR_ORDER,
    map_k1_tour, map_sa_tour, map_tour_for_kiviat,
)
from utils.ui import athlete_label_html
from utils.data_helpers import build_compet_label, get_compet_chrono_rank
from utils.interpretations import show_chart_guide
from utils.lang import t, get_lang


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Athlete info cards
# ═══════════════════════════════════════════════════════════════════════════════

def _athlete_info_block(df: pd.DataFrame, name: str):
    """Render a markdown block with key athlete metadata."""
    st.markdown(athlete_label_html(name), unsafe_allow_html=True)

    sexe = df["Sexe"].mode().iloc[0] if not df["Sexe"].mode().empty else t("Non spécifié")
    nation = df["Nation"].mode().iloc[0] if not df["Nation"].mode().empty else t("Non spécifié")
    style = df["Style"].mode().iloc[0] if not df["Style"].mode().empty else t("Non spécifié")

    # Sort by chronological order to get the most recent entry
    df_sorted = df.copy()
    df_sorted["_chrono"] = df_sorted.apply(
        lambda r: get_compet_chrono_rank(r.get("Competition", ""), r.get("Year")), axis=1
    )
    df_sorted = df_sorted.sort_values("_chrono", ascending=False)

    # Take the last known age from the most recent competition
    age_series = df_sorted["Age"].dropna()
    if age_series.empty:
        age_display = t("Non spécifié")
    else:
        try:
            age_display = f"{float(age_series.iloc[0]):.0f} {t('ans')}"
        except Exception:
            age_display = f"{age_series.iloc[0]} {t('ans')}"

    # Take the last known ranking from the most recent competition
    ranking_series = df_sorted["Ranking"].dropna()
    if ranking_series.empty:
        ranking_str = t("Non spécifié")
    else:
        try:
            ranking_str = str(int(ranking_series.iloc[0]))
        except Exception:
            ranking_str = str(ranking_series.iloc[0])

    st.markdown(
        f"""
        - **{t('Sexe')} :** {sexe}
        - **{t('Âge (dernier connu)')} :** {age_display}
        - **Ranking ({t('dernier connu')}) :** {ranking_str}
        - **{t('Nationalité')} :** {nation}
        - **{t('Style')} :** {style}
        """
    )


def render_athlete_info(state):
    st.subheader(t("Informations Athlète(s)"))
    col_left, col_sep, col_right = st.columns([1, 0.05, 1])

    with col_left:
        _athlete_info_block(state.athlete_data, state.selected_athlete)

    with col_sep:
        if state.compare_athlete_data is not None and not state.compare_athlete_data.empty:
            st.markdown("<div style='border-left:1px solid #d0d0d0;height:100%;'></div>", unsafe_allow_html=True)

    with col_right:
        if state.compare_athlete_data is not None and not state.compare_athlete_data.empty:
            _athlete_info_block(state.compare_athlete_data, state.selected_compare_athlete)
        else:
            st.markdown(
                f"<p style='font-size:12px;color:gray;'>{t('Aucun athlète comparé sélectionné')}</p>",
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Tour maximal atteint par compétition
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_max_tour_by_compet(df_src, athlete_name, type_compet):
    df_a = df_src[(df_src["Nom"] == athlete_name) & (df_src["Type_Compet"] == type_compet)].copy()
    if df_a.empty:
        return pd.DataFrame(columns=["Competition_Label", "Tour_Label", "Level", "Athlète"])

    df_a["Competition_Label"] = df_a.apply(
        lambda r: build_compet_label(r["Competition"], r.get("Year")), axis=1,
    )
    df_a["Year_numeric"] = pd.to_numeric(df_a.get("Year"), errors="coerce") if "Year" in df_a.columns else np.nan

    if type_compet == "K1":
        df_a["Tour_Label"] = df_a.apply(lambda r: map_k1_tour(r["N_Tour"], r.get("Victoire")), axis=1)
        df_a["Level"] = df_a["Tour_Label"].map(K1_TOUR_ORDER)
    else:
        df_a["Tour_Label"] = df_a.apply(lambda r: map_sa_tour(r["N_Tour"], r.get("Victoire")), axis=1)
        df_a["Level"] = df_a["Tour_Label"].map(SA_TOUR_ORDER)

    df_a = df_a.dropna(subset=["Competition_Label", "Tour_Label", "Level"])
    if df_a.empty:
        return pd.DataFrame(columns=["Competition_Label", "Tour_Label", "Level", "Athlète"])

    idx = df_a.groupby("Competition_Label")["Level"].idxmax()
    out = df_a.loc[idx, ["Competition_Label", "Tour_Label", "Level", "Year_numeric"]].copy()
    out["Athlète"] = athlete_name
    return out.sort_values(["Year_numeric", "Competition_Label"], na_position="first")


def _plot_max_tour(df_plot, title, key_suffix):
    if df_plot.empty:
        st.info(f"{t('Aucune donnée')} {title} {t('pour les athlètes sélectionnés.')}")
        return

    cat_order = df_plot.sort_values(
        ["Year_numeric", "Competition_Label"], na_position="first"
    )["Competition_Label"].unique()

    unique_levels = sorted(df_plot["Level"].unique())
    ticktexts = []
    for lvl in unique_levels:
        labels = sorted(df_plot.loc[df_plot["Level"] == lvl, "Tour_Label"].unique())
        ticktexts.append(" / ".join(labels))

    fig = px.bar(
        df_plot,
        x="Competition_Label", y="Level", color="Athlète",
        barmode="group", text="Tour_Label",
        labels={"Competition_Label": t("Compétition"), "Level": t("Tour maximal")},
        category_orders={"Competition_Label": list(cat_order)},
    )
    fig.update_traces(textposition="outside")
    fig.update_yaxes(tickmode="array", tickvals=unique_levels, ticktext=ticktexts)
    st.plotly_chart(fig, width="stretch", key=key_suffix)


def render_max_tour(state):
    st.subheader(t("Tour maximal atteint par compétition"))
    col_left, col_sep, col_right = st.columns([1, 0.05, 1])
    s = state
    df = s.full_data

    # K1
    df_k1 = _compute_max_tour_by_compet(df, s.selected_athlete, "K1")
    if s.compare_athlete_data is not None and not s.compare_athlete_data.empty:
        df_k1 = pd.concat([df_k1, _compute_max_tour_by_compet(df, s.selected_compare_athlete, "K1")], ignore_index=True)

    # SA
    df_sa = _compute_max_tour_by_compet(df, s.selected_athlete, "SA")
    if s.compare_athlete_data is not None and not s.compare_athlete_data.empty:
        df_sa = pd.concat([df_sa, _compute_max_tour_by_compet(df, s.selected_compare_athlete, "SA")], ignore_index=True)

    with col_left:
        st.markdown("### Premier League (K1)")
        _plot_max_tour(df_k1, "K1", f"max_tour_k1_{s.selected_sexe}_{s.selected_athlete}_{s.selected_compare_athlete}")

    with col_sep:
        st.markdown("<div style='border-left:1px solid #d0d0d0;height:100%;'></div>", unsafe_allow_html=True)

    with col_right:
        st.markdown("### Series A (SA)")
        _plot_max_tour(df_sa, "SA", f"max_tour_sa_{s.selected_sexe}_{s.selected_athlete}_{s.selected_compare_athlete}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Histogramme des Katas effectués
# ═══════════════════════════════════════════════════════════════════════════════

def _kata_histogram(df_source, tours, athlete_name, key_suffix):
    st.markdown(athlete_label_html(athlete_name), unsafe_allow_html=True)

    kata_data = df_source[df_source["N_Tour"].isin(tours)] if tours else df_source.copy()
    kata_counts = kata_data["Kata"].value_counts().reset_index()
    kata_counts.columns = ["Kata", "Nombre"]
    kata_counts = kata_counts[kata_counts["Nombre"] > 0]

    if kata_counts.empty:
        st.warning(t("Aucun Kata à afficher pour les tours sélectionnés."))
        return

    fig = px.bar(
        kata_counts, x="Kata", y="Nombre",
        title=t("Nombre de Katas effectués"),
        labels={"Nombre": t("Nombre de fois")}, text="Nombre",
    )
    fig.update_layout(xaxis_title="Kata", yaxis_title=t("Nombre de fois"))
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, width="stretch", key=key_suffix)


def render_kata_histogram(state):
    st.subheader(t("Histogramme des Katas effectués"))
    s = state
    col_left, col_sep, col_right = st.columns([1, 0.05, 1])

    with col_left:
        _kata_histogram(
            s.athlete_data, s.selected_tours, s.selected_athlete,
            f"hist_kata_main_{s.selected_sexe}_{s.selected_athlete}_{len(s.selected_tours)}",
        )

    with col_sep:
        if s.compare_athlete_data is not None and not s.compare_athlete_data.empty:
            st.markdown("<div style='border-left:1px solid #d0d0d0;height:100%;'></div>", unsafe_allow_html=True)

    with col_right:
        if s.compare_athlete_data is not None and not s.compare_athlete_data.empty:
            _kata_histogram(
                s.compare_athlete_data, s.selected_tours, s.selected_compare_athlete,
                f"hist_kata_comp_{s.selected_sexe}_{s.selected_compare_athlete}_{s.selected_athlete}_{len(s.selected_tours)}",
            )
        else:
            st.markdown(
                f"<p style='font-size:12px;color:gray;'>{t('Aucun athlète comparé sélectionné')}</p>",
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Kiviat – Moyenne des notes par Tour
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_avg_notes_by_tour(df_source, athlete_name, selected_competitions):
    note_data = df_source.copy()
    if selected_competitions:
        note_data = note_data[note_data["Competition"].isin(selected_competitions)]
    note_data = note_data.dropna(subset=["Note", "N_Tour"])
    if note_data.empty:
        return pd.DataFrame(columns=["Tour", "Moyenne_Note", "Athlète"])

    note_data["Tour_Kiviat"] = note_data["N_Tour"].apply(map_tour_for_kiviat)
    note_data = note_data.dropna(subset=["Tour_Kiviat"])
    if note_data.empty:
        return pd.DataFrame(columns=["Tour", "Moyenne_Note", "Athlète"])

    grouped = note_data.groupby("Tour_Kiviat")["Note"].mean().reset_index()
    grouped["Athlète"] = athlete_name
    grouped["Order"] = grouped["Tour_Kiviat"].map(KIVIAT_TOUR_ORDER).fillna(999)
    grouped = grouped.sort_values("Order")
    grouped.rename(columns={"Tour_Kiviat": "Tour", "Note": "Moyenne_Note"}, inplace=True)
    return grouped[["Tour", "Moyenne_Note", "Athlète"]]


def render_kiviat_tour(state):
    st.subheader(t("Moyenne des notes par Tour"))
    show_chart_guide("radar")
    s = state

    # For Kiviat, only use data with Notes (2024-2025). 2026 flag data not applicable for note averaging.
    athlete_data_notes = s.athlete_data[s.athlete_data["Note"].notna()] if "Note" in s.athlete_data.columns else s.athlete_data

    df_kiviat = _compute_avg_notes_by_tour(athlete_data_notes, s.selected_athlete, s.selected_competitions)
    if s.compare_athlete_data is not None and not s.compare_athlete_data.empty:
        compare_data_notes = s.compare_athlete_data[s.compare_athlete_data["Note"].notna()] if "Note" in s.compare_athlete_data.columns else s.compare_athlete_data
        df_comp = _compute_avg_notes_by_tour(compare_data_notes, s.selected_compare_athlete, s.selected_competitions)
        df_kiviat = pd.concat([df_kiviat, df_comp], ignore_index=True)

    if df_kiviat.empty:
        st.info(t("Aucune note disponible pour construire le diagramme par tour."))
        st.caption(t("Les données 2026+ utilisent le système de drapeaux (pas de notes)."))
        return

    # Dynamic range
    note_min = max(0.0, df_kiviat["Moyenne_Note"].min() - 2.0)
    note_max = min(50.0, df_kiviat["Moyenne_Note"].max() + 2.0)

    fig = go.Figure()
    for athlete_name in df_kiviat["Athlète"].unique():
        subset = df_kiviat[df_kiviat["Athlète"] == athlete_name]
        fig.add_trace(go.Scatterpolar(r=subset["Moyenne_Note"], theta=subset["Tour"], fill="toself", name=athlete_name))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[note_min, note_max])),
        showlegend=True,
        title=t("Moyenne des notes par tour (tours réellement disputés)"),
    )
    st.plotly_chart(
        fig, width="stretch",
        key=f"kiviat_tour_{s.selected_sexe}_{s.selected_athlete}_{s.selected_compare_athlete}_{len(s.selected_competitions)}",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Kiviat – Moyenne des notes par Kata
# ═══════════════════════════════════════════════════════════════════════════════

def _get_katas_for_athlete(df_source, selected_competitions):
    d = df_source.copy()
    if selected_competitions:
        d = d[d["Competition"].isin(selected_competitions)]
    return set(d["Kata"].dropna().unique())


def _compute_avg_notes_by_kata(df_source, athlete_name, kata_list, selected_competitions):
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


def render_kiviat_kata(state):
    st.subheader(t("Moyenne des notes par Kata"))
    show_chart_guide("radar")
    s = state

    katas_a = _get_katas_for_athlete(s.athlete_data, s.selected_competitions)
    if s.compare_athlete_data is not None and not s.compare_athlete_data.empty:
        katas_b = _get_katas_for_athlete(s.compare_athlete_data, s.selected_competitions)
        common = katas_a & katas_b
        kata_list = sorted(common) if common else sorted(katas_a | katas_b)
    else:
        kata_list = sorted(katas_a)

    df_kk = _compute_avg_notes_by_kata(s.athlete_data, s.selected_athlete, kata_list, s.selected_competitions)
    if s.compare_athlete_data is not None and not s.compare_athlete_data.empty:
        df_kk_comp = _compute_avg_notes_by_kata(s.compare_athlete_data, s.selected_compare_athlete, kata_list, s.selected_competitions)
        df_kk = pd.concat([df_kk, df_kk_comp], ignore_index=True)

    if df_kk.empty:
        st.info(t("Aucune note disponible pour construire le diagramme par Kata."))
        return

    df_kk["Kata"] = pd.Categorical(df_kk["Kata"], categories=kata_list, ordered=True)
    df_kk = df_kk.sort_values("Kata")

    note_min = max(0.0, df_kk["Moyenne_Note"].min() - 2.0)
    note_max = min(50.0, df_kk["Moyenne_Note"].max() + 2.0)

    fig = go.Figure()
    for athlete_name in df_kk["Athlète"].unique():
        subset = df_kk[df_kk["Athlète"] == athlete_name].copy()
        r_vals = subset["Moyenne_Note"].astype(float).round(2).tolist()
        theta_vals = subset["Kata"].astype(str).tolist()
        fig.add_trace(
            go.Scatterpolar(
                r=r_vals,
                theta=theta_vals,
                fill="toself",
                mode="lines+markers",
                marker=dict(size=7),
                name=athlete_name,
                text=[f"{v:.2f}" for v in r_vals],
                hovertemplate="<b>%{theta}</b><br>" + t("Note moyenne") + " : %{text}<extra>" + athlete_name + "</extra>",
            )
        )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[note_min, note_max])),
        showlegend=True,
        title=t("Moyenne des notes par Kata"),
    )
    st.plotly_chart(
        fig, width="stretch",
        key=f"kiviat_kata_{s.selected_sexe}_{s.selected_athlete}_{s.selected_compare_athlete}_{len(s.selected_competitions)}",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Flag-era radar – Win Rate & Drapeaux par Tour (2026+)
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_flag_stats_by_tour(df_source, athlete_name, selected_competitions):
    """Compute win rate and avg flags by tour for flag-era data."""
    d = df_source.copy()
    if selected_competitions:
        d = d[d["Competition"].isin(selected_competitions)]
    d = d.dropna(subset=["N_Tour"])
    # Flag era: Drapeau is present
    if "Drapeau" not in d.columns or d["Drapeau"].notna().sum() == 0:
        return pd.DataFrame()

    d = d[d["Drapeau"].notna()].copy()
    if d.empty:
        return pd.DataFrame()

    d["Win"] = d["Victoire"].astype(str).str.lower().isin(["true", "1", "vrai", "yes"]).astype(int)
    d["Tour_Kiviat"] = d["N_Tour"].apply(map_tour_for_kiviat)
    d = d.dropna(subset=["Tour_Kiviat"])
    if d.empty:
        return pd.DataFrame()

    grouped = d.groupby("Tour_Kiviat").agg(
        Win_Rate=("Win", "mean"),
        Drapeaux_Moy=("Drapeau", "mean"),
        Passages=("Win", "count"),
    ).reset_index()
    grouped["Win_Rate"] = (grouped["Win_Rate"] * 100).round(1)
    grouped["Drapeaux_Moy"] = grouped["Drapeaux_Moy"].round(2)
    grouped["Athlète"] = athlete_name
    grouped["Order"] = grouped["Tour_Kiviat"].map(KIVIAT_TOUR_ORDER).fillna(999)
    grouped = grouped.sort_values("Order")
    grouped.rename(columns={"Tour_Kiviat": "Tour"}, inplace=True)
    return grouped


def _compute_flag_stats_by_kata(df_source, athlete_name, kata_list, selected_competitions):
    """Compute win rate and avg flags by kata for flag-era data."""
    d = df_source.copy()
    if selected_competitions:
        d = d[d["Competition"].isin(selected_competitions)]
    if "Drapeau" not in d.columns or d["Drapeau"].notna().sum() == 0:
        return pd.DataFrame()

    d = d[d["Drapeau"].notna()].copy()
    d = d.dropna(subset=["Kata"])
    if kata_list:
        d = d[d["Kata"].isin(kata_list)]
    if d.empty:
        return pd.DataFrame()

    d["Win"] = d["Victoire"].astype(str).str.lower().isin(["true", "1", "vrai", "yes"]).astype(int)

    grouped = d.groupby("Kata").agg(
        Win_Rate=("Win", "mean"),
        Drapeaux_Moy=("Drapeau", "mean"),
        Passages=("Win", "count"),
    ).reset_index()
    grouped["Win_Rate"] = (grouped["Win_Rate"] * 100).round(1)
    grouped["Drapeaux_Moy"] = grouped["Drapeaux_Moy"].round(2)
    grouped["Athlète"] = athlete_name
    return grouped


def render_flag_radar_tour(state):
    """Radar chart showing flag-era performance by tour."""
    s = state
    athlete_flag_data = s.athlete_data[s.athlete_data["Drapeau"].notna()] if "Drapeau" in s.athlete_data.columns else pd.DataFrame()
    if athlete_flag_data.empty:
        return

    st.subheader(t("Performance drapeaux par Tour (2026+)"))

    df_flag = _compute_flag_stats_by_tour(s.athlete_data, s.selected_athlete, s.selected_competitions)
    if s.compare_athlete_data is not None and not s.compare_athlete_data.empty:
        df_flag_comp = _compute_flag_stats_by_tour(s.compare_athlete_data, s.selected_compare_athlete, s.selected_competitions)
        if not df_flag_comp.empty:
            df_flag = pd.concat([df_flag, df_flag_comp], ignore_index=True)

    if df_flag.empty:
        st.info(t("Aucune donnée drapeaux disponible."))
        return

    col_wr, col_dr = st.columns(2)

    with col_wr:
        fig_wr = go.Figure()
        for ath in df_flag["Athlète"].unique():
            subset = df_flag[df_flag["Athlète"] == ath]
            fig_wr.add_trace(go.Scatterpolar(
                r=subset["Win_Rate"], theta=subset["Tour"], fill="toself",
                name=ath,
                text=[f"{v:.0f}%" for v in subset["Win_Rate"]],
                hovertemplate="<b>%{theta}</b><br>Win rate : %{text}<extra>" + ath + "</extra>",
            ))
        fig_wr.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            title=t("Win rate par tour (drapeaux)"),
            height=380,
        )
        st.plotly_chart(fig_wr, width="stretch", key=f"flag_wr_tour_{s.selected_athlete}_{s.selected_compare_athlete}")

    with col_dr:
        # Determine max drapeaux based on competition type
        max_flag = df_flag["Drapeaux_Moy"].max() + 1 if not df_flag.empty else 7
        fig_dr = go.Figure()
        for ath in df_flag["Athlète"].unique():
            subset = df_flag[df_flag["Athlète"] == ath]
            fig_dr.add_trace(go.Scatterpolar(
                r=subset["Drapeaux_Moy"], theta=subset["Tour"], fill="toself",
                name=ath,
                text=[f"{v:.1f}" for v in subset["Drapeaux_Moy"]],
                hovertemplate="<b>%{theta}</b><br>" + t("Drapeau moy.") + " : %{text}<extra>" + ath + "</extra>",
            ))
        fig_dr.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, max_flag])),
            showlegend=True,
            title=t("Drapeaux moyens par tour"),
            height=380,
        )
        st.plotly_chart(fig_dr, width="stretch", key=f"flag_dr_tour_{s.selected_athlete}_{s.selected_compare_athlete}")


def render_flag_radar_kata(state):
    """Radar chart showing flag-era performance by kata."""
    s = state
    athlete_flag_data = s.athlete_data[s.athlete_data["Drapeau"].notna()] if "Drapeau" in s.athlete_data.columns else pd.DataFrame()
    if athlete_flag_data.empty:
        return

    st.subheader(t("Performance drapeaux par Kata (2026+)"))

    # Only katas from athlete's full history (not just flag era)
    kata_list = sorted(s.athlete_data["Kata"].dropna().unique())

    df_fk = _compute_flag_stats_by_kata(s.athlete_data, s.selected_athlete, kata_list, s.selected_competitions)
    if s.compare_athlete_data is not None and not s.compare_athlete_data.empty:
        df_fk_comp = _compute_flag_stats_by_kata(s.compare_athlete_data, s.selected_compare_athlete, kata_list, s.selected_competitions)
        if not df_fk_comp.empty:
            df_fk = pd.concat([df_fk, df_fk_comp], ignore_index=True)

    if df_fk.empty:
        st.info(t("Aucune donnée drapeaux par kata disponible."))
        return

    col_wr, col_dr = st.columns(2)

    with col_wr:
        fig_wr = go.Figure()
        for ath in df_fk["Athlète"].unique():
            subset = df_fk[df_fk["Athlète"] == ath]
            fig_wr.add_trace(go.Scatterpolar(
                r=subset["Win_Rate"], theta=subset["Kata"].astype(str), fill="toself",
                name=ath,
                text=[f"{v:.0f}%" for v in subset["Win_Rate"]],
                hovertemplate="<b>%{theta}</b><br>Win rate : %{text}<extra>" + ath + "</extra>",
            ))
        fig_wr.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            title=t("Win rate par kata (drapeaux)"),
            height=380,
        )
        st.plotly_chart(fig_wr, width="stretch", key=f"flag_wr_kata_{s.selected_athlete}_{s.selected_compare_athlete}")

    with col_dr:
        max_flag = df_fk["Drapeaux_Moy"].max() + 1 if not df_fk.empty else 7
        fig_dr = go.Figure()
        for ath in df_fk["Athlète"].unique():
            subset = df_fk[df_fk["Athlète"] == ath]
            fig_dr.add_trace(go.Scatterpolar(
                r=subset["Drapeaux_Moy"], theta=subset["Kata"].astype(str), fill="toself",
                name=ath,
                text=[f"{v:.1f}" for v in subset["Drapeaux_Moy"]],
                hovertemplate="<b>%{theta}</b><br>" + t("Drapeau moy.") + " : %{text}<extra>" + ath + "</extra>",
            ))
        fig_dr.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, max_flag])),
            showlegend=True,
            title=t("Drapeaux moyens par kata"),
            height=380,
        )
        st.plotly_chart(fig_dr, width="stretch", key=f"flag_dr_kata_{s.selected_athlete}_{s.selected_compare_athlete}")
