"""Tour mappings and classification helpers for K1 and SA competitions."""

import pandas as pd

# ── K1 (Premier League) ──────────────────────────────────────────────────────

K1_TOUR_LABEL = {
    "Pool_1": "Poule",
    "Pool_2": "Poule",
    "Pool_3": "Poule",
    "R1": "Quart de finale",
    "R2": "Demi finale",
    "Finale": "Finale",
    "Final": "Finale",
}

K1_TOUR_ORDER = {
    "Poule": 1,
    "Quart de finale": 2,
    "Demi finale": 3,
    "Place de 3": 4,
    "Bronze": 4,
    "Finale": 5,
}

# ── SA (Series A) ────────────────────────────────────────────────────────────

SA_TOUR_LABEL = {
    "T1": "1er Tour",
    "T2": "2ème Tour",
    "T3": "3ème Tour",
    "R1": "Quart de finale",
    "R2": "Demi finale",
    "Finale": "Finale",
    "Final": "Finale",
    "PW1": "Finale de poule",
    "PW2": "Finale de poule",
    "PW3": "Finale de poule",
    "RP1": "1er Tour repêchage",
    "RP2": "2ème Tour repêchage",
    "RP3": "3ème Tour repêchage",
    "RP4": "4ème Tour repêchage",
}

SA_TOUR_ORDER = {
    "1er Tour": 1,
    "2ème Tour": 2,
    "3ème Tour": 3,
    "Finale de poule": 4,
    "1er Tour repêchage": 5,
    "2ème Tour repêchage": 6,
    "3ème Tour repêchage": 7,
    "4ème Tour repêchage": 8,
    "Quart de finale": 9,
    "Demi finale": 10,
    "Place de 3": 11,
    "Bronze": 11,
    "Finale": 12,
}

# ── Kiviat (radar charts) ────────────────────────────────────────────────────

KIVIAT_TOUR_MAP = {
    "Pool_1": "Poule",
    "Pool_2": "Poule",
    "Pool_3": "Poule",
    "R1": "Quart de finale",
    "R2": "Demi finale",
    "Bronze": "Bronze / Place de 3",
    "Finale": "Finale",
    "Final": "Finale",
    "T1": "1er Tour",
    "T2": "2ème Tour",
    "T3": "3ème Tour",
    "PW1": "Finale de poule",
    "PW2": "Finale de poule",
    "PW3": "Finale de poule",
    "RP1": "Repêchage",
    "RP2": "Repêchage",
    "RP3": "Repêchage",
    "RP4": "Repêchage",
}

KIVIAT_TOUR_ORDER = {
    "Poule": 1,
    "1er Tour": 2,
    "2ème Tour": 3,
    "3ème Tour": 4,
    "Finale de poule": 5,
    "Repêchage": 6,
    "Quart de finale": 7,
    "Demi finale": 8,
    "Bronze / Place de 3": 9,
    "Finale": 10,
}

# ── Tour type classification (ACM) ───────────────────────────────────────────

TOUR_TYPE_MAP = {
    "Bronze": "Match de médaille",
    "Finale": "Match de médaille",
    "Final": "Match de médaille",
    "R1": "Quart/Demi/Finale de poule",
    "R2": "Quart/Demi/Finale de poule",
    "PW1": "Quart/Demi/Finale de poule",
    "PW2": "Quart/Demi/Finale de poule",
    "PW3": "Quart/Demi/Finale de poule",
    "RP1": "Repêchage",
    "RP2": "Repêchage",
    "RP3": "Repêchage",
    "RP4": "Repêchage",
    "Pool_1": "Tour 1",
    "T1": "Tour 1",
    "Pool_2": "Tour 2",
    "T2": "Tour 2",
    "Pool_3": "Tour 3",
    "T3": "Tour 3",
}


# ── Helper functions ──────────────────────────────────────────────────────────

def map_k1_tour(n_tour, victoire=None):
    """Map raw N_Tour value to K1 tour label."""
    if pd.isna(n_tour):
        return None
    n_tour = str(n_tour)
    if n_tour == "Bronze":
        if pd.notna(victoire) and bool(victoire):
            return "Bronze"
        return "Place de 3"
    return K1_TOUR_LABEL.get(n_tour)


def map_sa_tour(n_tour, victoire=None):
    """Map raw N_Tour value to SA tour label."""
    if pd.isna(n_tour):
        return None
    n_tour = str(n_tour)
    if n_tour == "Bronze":
        if pd.notna(victoire) and bool(victoire):
            return "Bronze"
        return "Place de 3"
    return SA_TOUR_LABEL.get(n_tour)


def map_tour_for_kiviat(n_tour):
    """Map raw N_Tour value to Kiviat radar chart label."""
    if pd.isna(n_tour):
        return None
    return KIVIAT_TOUR_MAP.get(str(n_tour))


def classify_tour_type(n_tour):
    """Classify a tour into a type category (for ACM)."""
    if pd.isna(n_tour):
        return None
    return TOUR_TYPE_MAP.get(str(n_tour))
