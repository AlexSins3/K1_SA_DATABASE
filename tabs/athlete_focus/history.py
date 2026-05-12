# tabs/athlete_focus/history.py

from __future__ import annotations

import pandas as pd
import streamlit as st

from utils.ui import highlight_victory_series
from utils.data_helpers import build_compet_label, victoire_to_str
from utils.display import fmt_tour
from utils.lang import t


def render_history(state):
    """Render section 6 – Historique (tours for single athlete, encounters for pair)."""
    st.subheader(t("Historique"))
    s = state

    df_hist = s.full_data.copy()

    if s.selected_type_compet == "Premier League (K1)":
        df_hist = df_hist[df_hist["Type_Compet"] == "K1"]
    elif s.selected_type_compet == "Series A (SA)":
        df_hist = df_hist[df_hist["Type_Compet"] == "SA"]

    if s.selected_sexe and "Sexe" in df_hist.columns:
        df_hist = df_hist[df_hist["Sexe"] == s.selected_sexe]

    if s.selected_competitions:
        df_hist = df_hist[df_hist["Competition"].isin(s.selected_competitions)]

    # Sort for robust R/B pairing
    df_hist = df_hist.sort_values(
        ["Competition", "Year", "Type_Compet", "N_Tour"], kind="mergesort",
    ).reset_index(drop=True)

    if s.compare_athlete_data is None or s.compare_athlete_data.empty:
        _render_single_history(df_hist, s.selected_athlete)
    else:
        _render_pair_history(df_hist, s.selected_athlete, s.selected_compare_athlete)


# ── Single athlete: tour-by-tour history ──────────────────────────────────────

def _render_single_history(df_hist, athlete_name):
    st.markdown(f"##### {t('Historique des tours')}")
    df_a = df_hist[df_hist["Nom"] == athlete_name].copy()

    if df_a.empty:
        st.info(t("Aucun historique trouvé pour cet athlète avec les filtres actuels."))
        return

    records = []
    for idx, row in df_a.iterrows():
        ceinture = row.get("Ceinture")
        opp_name = "Inconnu"

        if ceinture == "R" and idx + 1 < len(df_hist):
            opp_row = df_hist.iloc[idx + 1]
            if opp_row.get("Ceinture") == "B" and _same_context(row, opp_row):
                opp_name = opp_row.get("Nom", "Inconnu")
        elif ceinture == "B" and idx - 1 >= 0:
            opp_row = df_hist.iloc[idx - 1]
            if opp_row.get("Ceinture") == "R" and _same_context(row, opp_row):
                opp_name = opp_row.get("Nom", "Inconnu")

        records.append({
            "Tour": fmt_tour(row.get("N_Tour")),
            "Kata": row.get("Kata"),
            "Note": row.get("Note"),
            "vs.": opp_name,
            "Victoire": victoire_to_str(row.get("Victoire")),
            "Competition": build_compet_label(row.get("Competition"), row.get("Year")),
        })

    hist_df = pd.DataFrame(records).sort_values(["Competition", "Tour"], na_position="last")
    styled = hist_df.style.apply(
        lambda s: highlight_victory_series(s) if s.name == "Victoire" else [""] * len(s), axis=0,
    )
    st.dataframe(styled, use_container_width=True)


# ── Pair of athletes: head-to-head history ────────────────────────────────────

def _render_pair_history(df_hist, athlete_a, athlete_b):
    st.markdown("##### Historique des rencontres")
    records = []
    n_rows = len(df_hist)

    for i in range(n_rows - 1):
        r1 = df_hist.iloc[i]
        r2 = df_hist.iloc[i + 1]

        noms = {r1.get("Nom"), r2.get("Nom")}
        ceintures = {str(r1.get("Ceinture")), str(r2.get("Ceinture"))}

        if {athlete_a, athlete_b}.issubset(noms) and {"R", "B"}.issubset(ceintures) and _same_context(r1, r2):
            self_row = r1 if r1.get("Nom") == athlete_a else r2
            records.append({
                "Tour": fmt_tour(self_row.get("N_Tour")),
                "Kata": self_row.get("Kata"),
                "Note": self_row.get("Note"),
                "Ceinture": self_row.get("Ceinture"),
                "Victoire": victoire_to_str(self_row.get("Victoire")),
                "Competition": build_compet_label(self_row.get("Competition"), self_row.get("Year")),
            })

    if not records:
        st.info("Les deux athlètes ne se sont pas encore affrontés (ou pas avec les filtres actuels).")
        return

    meet_df = pd.DataFrame(records).sort_values(["Competition", "Tour"], na_position="last")
    styled = meet_df.style.apply(
        lambda s: highlight_victory_series(s) if s.name == "Victoire" else [""] * len(s), axis=0,
    )
    st.dataframe(styled, use_container_width=True)


def _same_context(r1, r2) -> bool:
    """Check that two rows belong to the same match context."""
    return (
        str(r1.get("Competition")) == str(r2.get("Competition"))
        and str(r1.get("Type_Compet")) == str(r2.get("Type_Compet"))
        and str(r1.get("N_Tour")) == str(r2.get("N_Tour"))
        and str(r1.get("Year")) == str(r2.get("Year"))
    )
