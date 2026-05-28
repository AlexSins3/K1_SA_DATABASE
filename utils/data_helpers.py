"""Shared data-transformation helpers."""

import numpy as np
import pandas as pd

from utils.lang import t


# ── Competition chronological order (for time-based weighting) ────────────────

COMPET_CHRONO_ORDER = {
    # 2024
    "K1 Paris": 1, "K1 Antalya": 2, "K1 Cairo": 3, "K1 Casablanca": 4,
    "SA Athens": 5, "SA Larnaca": 6, "SA Salzbourg": 7,
    # 2025
    "K1 Hanghou": 8, "K1 Rabat": 9,
    "SA KualaLumpur": 10, "SA Kuala Lumpur": 10,
    # 2026
    "K1 Istanbul": 11, "SA Tbilisi": 12, "K1 Roma": 13,
    "K1 Leshan": 14, "SA ACoruna": 15, "SA A Coruna": 15,
}

# Max flags per judge system
MAX_FLAGS_SA = 5   # 5 judges for Series A
MAX_FLAGS_K1 = 7   # 7 judges for Premier League


def get_compet_chrono_rank(competition: str, year=None) -> int:
    """Return chronological rank of a competition (higher = more recent)."""
    comp_str = str(competition).replace("_", " ") if competition else ""
    rank = COMPET_CHRONO_ORDER.get(comp_str, 0)
    if rank == 0 and year is not None:
        try:
            rank = int(year) * 10
        except (ValueError, TypeError):
            pass
    return rank


def compute_time_weight(competition: str, year=None, decay: float = 0.85) -> float:
    """Compute exponential time weight — more recent competitions get higher weight.
    
    decay=0.85 means each step back in time multiplies weight by 0.85.
    Most recent competition gets weight ~1.0.
    """
    rank = get_compet_chrono_rank(competition, year)
    max_rank = max(COMPET_CHRONO_ORDER.values()) if COMPET_CHRONO_ORDER else 1
    steps_back = max_rank - rank
    return decay ** steps_back


def is_flag_era(year) -> bool:
    """Return True if the year uses the flag judging system (2026+)."""
    try:
        return int(year) >= 2026
    except (ValueError, TypeError):
        return False


def classify_match_flags(flags_winner: int, flags_loser: int, type_compet: str) -> str:
    """Classify match type based on flag differential.
    
    SA (5 judges): Close=3-2, Net=4-1, Unanimous=5-0
    K1 (7 judges): Close=4-3, Net=5-2/6-1, Unanimous=7-0
    """
    diff = abs(flags_winner - flags_loser)
    if type_compet == "SA":
        if diff <= 1:  # 3-2
            return "Serré (1 drapeau)"
        elif diff <= 3:  # 4-1
            return "Net (2-3 drapeaux)"
        else:  # 5-0
            return "Unanime"
    else:  # K1
        if diff <= 1:  # 4-3
            return "Serré (1 drapeau)"
        elif diff <= 3:  # 5-2, 6-1
            return "Net (2-3 drapeaux)"
        else:  # 7-0
            return "Unanime"


def classify_match_notes(margin: float) -> str:
    """Classify match type based on note margin (2024-2025 system)."""
    if margin < 0.5:
        return "Serrée (<0.5)"
    elif margin <= 1.5:
        return "Nette (0.5-1.5)"
    return "Dominante (>1.5)"


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
    """Return ``'Oui'`` / ``'Non'`` / ``'Inconnu'`` (translated)."""
    b = victoire_to_bool(v)
    if b is True:
        return t("Oui")
    if b is False:
        return t("Non")
    return t("Inconnu")


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
