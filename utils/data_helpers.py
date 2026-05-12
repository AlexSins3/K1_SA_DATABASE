"""Shared data-transformation helpers."""

import numpy as np
import pandas as pd


# ── Victoire conversion (single source of truth) ─────────────────────────────

_TRUE_TOKENS = {"true", "vrai", "1", "oui", "o", "y", "yes", "win", "gagné", "gagne"}
_FALSE_TOKENS = {"false", "faux", "0", "non", "n", "no", "lose", "perdu"}


def victoire_to_bool(v):
    """Convert any raw *Victoire* value to ``True`` / ``False`` / ``None``."""
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    if pd.isna(v):
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    try:
        nv = float(v)
        if nv == 1:
            return True
        if nv == 0:
            return False
    except (ValueError, TypeError):
        pass
    s = str(v).strip().lower()
    if s in _TRUE_TOKENS:
        return True
    if s in _FALSE_TOKENS:
        return False
    return None


def victoire_to_str(v) -> str:
    """Return ``'Oui'`` / ``'Non'`` / ``'Inconnu'``."""
    b = victoire_to_bool(v)
    if b is True:
        return "Oui"
    if b is False:
        return "Non"
    return "Inconnu"


def victoire_to_int(v):
    """Return ``1`` / ``0`` / ``None``."""
    b = victoire_to_bool(v)
    if b is None:
        return None
    return int(b)


def normalize_victoire_category(v):
    """Return ``'True'`` / ``'False'`` or ``np.nan`` (for ACM variables)."""
    b = victoire_to_bool(v)
    if b is True:
        return "True"
    if b is False:
        return "False"
    return np.nan


# ── Misc helpers ──────────────────────────────────────────────────────────────

def build_compet_label(competition, year=None) -> str:
    """Build a ``Competition_Year`` label string."""
    comp = str(competition) if competition is not None else ""
    if year is not None and not pd.isna(year):
        try:
            return f"{comp}_{int(year)}"
        except (ValueError, TypeError):
            return comp
    return comp


def safe_mode(series: pd.Series, default="Non spécifié"):
    """Return the statistical mode of *series*, with a fallback *default*."""
    m = series.dropna()
    if m.empty:
        return default
    try:
        return m.mode().iloc[0]
    except Exception:
        return m.iloc[0]


def get_numeric_and_categorical_columns(df: pd.DataFrame):
    """Split *df* columns into ``(numeric_cols, categorical_cols)``."""
    numeric_cols = [
        col for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].dtype != bool
    ]
    categorical_cols = [col for col in df.columns if col not in numeric_cols]
    return numeric_cols, categorical_cols
