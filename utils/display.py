"""Display formatting for coach-friendly presentation — visual only, no data mutation."""

from __future__ import annotations

import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════════
# Tour display mapping
# ═══════════════════════════════════════════════════════════════════════════════

TOUR_DISPLAY = {
    "T1": "1er tour Series A",
    "T2": "2ème tour Series A",
    "T3": "3ème tour Series A",
    "Pool_1": "1er tour de poule K1",
    "Pool_2": "2ème tour de poule K1",
    "Pool_3": "3ème tour de poule K1",
    "PW1": "8ème de finale Series A",
    "PW2": "Quart de finale Series A",
    "PW3": "Demi-finale Series A",
    "R1": "Quart de finale K1",
    "R2": "Demi-finale K1",
    "Bronze": "Match pour la médaille de bronze",
    "Final": "Finale",
    "Finale": "Finale",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Formatting functions
# ═══════════════════════════════════════════════════════════════════════════════

def fmt_athlete(name) -> str:
    """Reformat 'Nom_Prenom' → 'Prenom Nom', 'Nom1_Nom2_Prenom' → 'Prenom Nom1 Nom2'."""
    try:
        if name is None or pd.isna(name):
            return ""
    except (TypeError, ValueError):
        pass
    s = str(name).strip()
    if not s or s == "nan" or s == "<NA>":
        return s
    parts = s.split("_")
    if len(parts) <= 1:
        return s
    prenom = parts[-1]
    nom = " ".join(parts[:-1])
    return f"{prenom} {nom}"


def fmt_tour(tour) -> str:
    """Map tour code to readable label for display."""
    try:
        if tour is None or pd.isna(tour):
            return ""
    except (TypeError, ValueError):
        pass
    return TOUR_DISPLAY.get(str(tour).strip(), str(tour).replace("_", " "))


def fmt_underscore(text) -> str:
    """Replace underscores with spaces for display."""
    try:
        if text is None or pd.isna(text):
            return ""
    except (TypeError, ValueError):
        pass
    return str(text).replace("_", " ")


def fmt_col(col_name: str) -> str:
    """Format a column header for display."""
    return str(col_name).replace("_", " ")


# ═══════════════════════════════════════════════════════════════════════════════
# DataFrame display helpers
# ═══════════════════════════════════════════════════════════════════════════════

_TOUR_COLS = {"N_Tour", "Tour", "Tour_Max", "N Tour", "Tour Max", "Tour_Max_Freq"}


def fmt_df(df: pd.DataFrame) -> pd.DataFrame:
    """Format tour values in a dataframe copy (keeps column names intact for plotly)."""
    out = df.copy()
    for col in list(out.columns):
        if col in _TOUR_COLS:
            out[col] = out[col].apply(fmt_tour)
    return out


def format_display_df(df: pd.DataFrame) -> pd.DataFrame:
    """Format a dataframe copy for st.dataframe(): tour values + column header rename."""
    out = fmt_df(df)
    out.columns = [fmt_col(c) for c in out.columns]
    return out
