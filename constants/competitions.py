"""Competition types, judging panels, chronology and tour ranks.

Pure-Python module (no Streamlit import) so it can be shared by the app
and by the training scripts in ``model/``.
See ``data/conventions_database.json`` for the naming conventions.
"""

# ── Competition types ─────────────────────────────────────────────────────────
# judges = number of flags per match in the flag era (2026+).

COMPET_TYPES = {
    "K1":   {"fr": "Premier League (K1)", "en": "Premier League (K1)", "judges": 7},
    "SA":   {"fr": "Series A (SA)", "en": "Series A (SA)", "judges": 5},
    "EKF":  {"fr": "Championnats d'Europe (EKF)", "en": "European Championships (EKF)", "judges": 7},
    "AKF":  {"fr": "Championnats d'Asie (AKF)", "en": "Asian Championships (AKF)", "judges": 7},
    "PKF":  {"fr": "Championnats panaméricains (PKF)", "en": "Pan American Championships (PKF)", "judges": 5},
    "UFAK": {"fr": "Championnats d'Afrique (UFAK)", "en": "African Championships (UFAK)", "judges": 7},
    "WKF":  {"fr": "Championnats du monde (WKF)", "en": "World Championships (WKF)", "judges": 7},
}

CIRCUIT_TYPES = ("K1", "SA")
CHAMPIONSHIP_TYPES = ("EKF", "AKF", "PKF", "UFAK", "WKF")


def is_championship(type_compet) -> bool:
    """True for continental / world championships (EKF, AKF, PKF, UFAK, WKF)."""
    return str(type_compet) in CHAMPIONSHIP_TYPES


def max_flags(type_compet) -> int:
    """Number of judges (flags per match) for a competition type."""
    return COMPET_TYPES.get(str(type_compet), {}).get("judges", 7)


def compet_type_label(type_compet, lang: str = "fr") -> str:
    """Readable label of a competition type (falls back to the raw code)."""
    info = COMPET_TYPES.get(str(type_compet))
    if info is None:
        return str(type_compet)
    return info["en"] if lang == "en" else info["fr"]


# ── Model groups by judging panel ────────────────────────────────────────────
# The flag-score models predict 0..N flags, so they are trained per panel size.

PANEL_GROUPS = {
    "J7": {"judges": 7, "types": ("K1", "EKF", "AKF", "UFAK", "WKF")},
    "J5": {"judges": 5, "types": ("SA", "PKF")},
}


def panel_group(type_compet) -> str:
    """Model group key ('J7' / 'J5') of a competition type."""
    return "J5" if max_flags(type_compet) == 5 else "J7"


# ── Chronological order ───────────────────────────────────────────────────────
# Keyed by (competition name with spaces, year). Higher = more recent.
# 2026 dates: Istanbul (Jan), Tbilisi (Feb), Roma (Mar), Leshan (10-12 Apr),
# A Coruna (24-26 Apr), EKF Frankfurt (20-24 May), PKF Rio (28-31 May),
# K1 Rabat (12-14 Jun), AKF Bali (19-21 Jun), UFAK Algiers (10-13 Sep),
# SA Salzbourg (2-4 Oct).

COMPET_CHRONO_ORDER = {
    # 2024
    ("K1 Paris", 2024): 1, ("K1 Antalya", 2024): 2, ("K1 Cairo", 2024): 3,
    ("K1 Casablanca", 2024): 4, ("SA Athens", 2024): 5, ("SA Larnaca", 2024): 6,
    ("SA Salzbourg", 2024): 7,
    # 2025
    ("K1 Hanghou", 2025): 8, ("K1 Rabat", 2025): 9, ("SA KualaLumpur", 2025): 10,
    ("SA Kuala Lumpur", 2025): 10, ("K1 Cairo", 2025): 11, ("K1 Paris", 2025): 12,
    ("SA Larnaca", 2025): 13, ("SA Salzbourg", 2025): 14, ("SA Tbilisi", 2025): 15,
    # 2026
    ("K1 Istanbul", 2026): 16, ("SA Tbilisi", 2026): 17, ("K1 Roma", 2026): 18,
    ("K1 Leshan", 2026): 19, ("SA ACoruna", 2026): 20, ("SA A Coruna", 2026): 20,
    ("EKF Frankfurt", 2026): 21, ("PKF Rio", 2026): 22, ("K1 Rabat", 2026): 23,
    ("AKF Bali", 2026): 24, ("UFAK Algiers", 2026): 25, ("SA Salzbourg", 2026): 26,
}


def get_compet_chrono_rank(competition, year=None) -> int:
    """Return chronological rank of a competition (higher = more recent).

    Uses (competition, year) lookup. Falls back to year-based estimation
    for competitions not explicitly listed (future-proof).
    """
    comp_str = str(competition).replace("_", " ") if competition else ""
    year_int = None
    try:
        year_int = int(year) if year is not None else None
    except (ValueError, TypeError):
        pass

    if year_int is None:
        return 0

    rank = COMPET_CHRONO_ORDER.get((comp_str, year_int), 0)
    if rank > 0:
        return rank

    # Fallback: place after all known competitions of that year
    year_ranks = [r for (_, y), r in COMPET_CHRONO_ORDER.items() if y == year_int]
    if year_ranks:
        return max(year_ranks)
    max_known_rank = max(COMPET_CHRONO_ORDER.values())
    max_known_year = max(y for (_, y) in COMPET_CHRONO_ORDER.keys())
    return max_known_rank + (year_int - max_known_year) * 7


# ── Tour rank (ordinal feature used by the models) ───────────────────────────
# SM (semi-final of championships) is the same stage as PW3 (SA semi-final).

TOUR_RANK = {
    "Pool_1": 1, "Pool_2": 2, "Pool_3": 3, "Pool_4": 4,
    "T1": 1, "T2": 2, "T3": 3,
    "PW1": 4, "PW2": 5, "PW3": 6, "SM": 6,
    "RP1": 4, "RP2": 5, "RP3": 6, "RP4": 7,
    "R1": 7, "R2": 8,
    "Bronze": 9, "Final": 10,
}
