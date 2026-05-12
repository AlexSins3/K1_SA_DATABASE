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
from utils.data_helpers import build_compet_label
from utils.interpretations import show_chart_guide
from utils.lang import t


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Athlete info cards
# ═══════════════════════════════════════════════════════════════════════════════

def _athlete_info_block(df: pd.DataFrame, name: str):
    """Render a markdown block with key athlete metadata."""
    st.markdown(athlete_label_html(name), unsafe_allow_html=True)

    sexe = df["Sexe"].mode().iloc[0] if not df["Sexe"].mode().empty else t("Non spécifié")
    nation = df["Nation"].mode().iloc[0] if not df["Nation"].mode().empty else t("Non spécifié")
    style = df["Style"].mode().iloc[0] if not df["Style"].mode().empty else t("Non spécifié")

    age_series = df["Age"].dropna()
    if age_series.empty:
        age_display = t("Non spécifié")
    else:
        try:
            age_display = f"{float(age_series.iloc[-1]):.1f} ans"
        except Exception:
            age_display = f"{age_series.iloc[-1]} ans"

    ranking_series = df["Ranking"].dropna()
    if ranking_series.empty:
        ranking_str = t("Non spécifié")
    else:
        try:
            ranking_str = str(int(ranking_series.iloc[-1]))
        except Exception:
            ranking_str = str(ranking_series.iloc[-1])

    st.markdown(
        f"""
        - **Sexe :** {sexe}
        - **Âge (dernier connu) :** {age_display}
        - **Ranking (dernier connu) :** {ranking_str}
        - **Nationalité :** {nation}
        - **Style :** {style}
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
    st.plotly_chart(fig, use_container_width=True, key=key_suffix)


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
    st.plotly_chart(fig, use_container_width=True, key=key_suffix)


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

    df_kiviat = _compute_avg_notes_by_tour(s.athlete_data, s.selected_athlete, s.selected_competitions)
    if s.compare_athlete_data is not None and not s.compare_athlete_data.empty:
        df_comp = _compute_avg_notes_by_tour(s.compare_athlete_data, s.selected_compare_athlete, s.selected_competitions)
        df_kiviat = pd.concat([df_kiviat, df_comp], ignore_index=True)

    if df_kiviat.empty:
        st.info(t("Aucune note disponible pour construire le diagramme par tour."))
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
        fig, use_container_width=True,
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
                hovertemplate="<b>%{theta}</b><br>Note moyenne : %{text}<extra>" + athlete_name + "</extra>",
            )
        )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[note_min, note_max])),
        showlegend=True,
        title=t("Moyenne des notes par Kata"),
    )
    st.plotly_chart(
        fig, use_container_width=True,
        key=f"kiviat_kata_{s.selected_sexe}_{s.selected_athlete}_{s.selected_compare_athlete}_{len(s.selected_competitions)}",
    )
