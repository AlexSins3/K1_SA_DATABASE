"""Display formatting for coach-friendly presentation — visual only, no data mutation."""

from __future__ import annotations

import pandas as pd

from utils.lang import t


# ═══════════════════════════════════════════════════════════════════════════════
# Tour display mapping
# ═══════════════════════════════════════════════════════════════════════════════

# Generic label of each N_Tour code (used when several formats are mixed).
TOUR_DISPLAY = {
    "T1": "1er tour",
    "T2": "2ème tour",
    "T3": "3ème tour",
    "Pool_1": "1er tour de poule K1",
    "Pool_2": "2ème tour de poule K1",
    "Pool_3": "3ème tour de poule K1",
    "PW1": "Finale de poule",
    "PW2": "Quart de finale Series A",
    "PW3": "Demi-finale Series A",
    "SM": "Demi-finale",
    "R1": "Quart de finale K1",
    "R2": "Demi-finale K1",
    "RP1": "1er tour de repêchage",
    "RP2": "2ème tour de repêchage",
    "RP3": "3ème tour de repêchage",
    "RP4": "4ème tour de repêchage",
    "Bronze": "Match pour la médaille de bronze",
    "Final": "Finale",
    "Finale": "Finale",
}

# Format-specific stage name of a code, suffixed with the competition name
# (e.g. PW1 = 8ème de finale in SA but Quart de finale in EKF).
_TOUR_STAGE_BY_TYPE = {
    "SA": {"T1": "1er tour", "T2": "2ème tour", "T3": "3ème tour", "PW1": "8ème de finale"},
    "EKF": {"T1": "1er tour", "T2": "2ème tour", "T3": "3ème tour", "PW1": "Quart de finale", "SM": "Demi-finale"},
    "WKF": {"T1": "1er tour", "T2": "2ème tour", "T3": "3ème tour", "PW1": "Quart de finale", "SM": "Demi-finale"},
    "AKF": {"T1": "1er tour", "T2": "2ème tour", "T3": "Quart de finale", "SM": "Demi-finale"},
    "PKF": {"T1": "1er tour", "T2": "2ème tour", "T3": "Quart de finale", "SM": "Demi-finale"},
    "UFAK": {"T1": "1er tour", "T2": "2ème tour", "T3": "Quart de finale", "SM": "Demi-finale"},
}
_TYPE_SUFFIX = {"SA": "Series A"}


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


def fmt_tour(tour, type_compet=None) -> str:
    """Map tour code to readable label for display (translated).

    With *type_compet* (K1, SA, EKF, AKF, PKF, UFAK…) the label names the
    actual stage of that format; without it a generic label is used.
    """
    try:
        if tour is None or pd.isna(tour):
            return ""
    except (TypeError, ValueError):
        pass
    code = str(tour).strip()
    stage = _TOUR_STAGE_BY_TYPE.get(str(type_compet), {}).get(code) if type_compet else None
    if stage is not None:
        return f"{t(stage)} {_TYPE_SUFFIX.get(str(type_compet), str(type_compet))}"
    return t(TOUR_DISPLAY.get(code, code.replace("_", " ")))


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
