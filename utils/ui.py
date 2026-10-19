"""Shared Streamlit UI components (CSS, filter panels, footer)."""

import html as html_lib
from dataclasses import dataclass

import pandas as pd
import streamlit as st

from constants.competitions import CHAMPIONSHIP_TYPES, compet_type_label
from utils.lang import t, get_lang

FILTER_PANEL_CSS = """
<style>
.filter-panel {
    background-color: #f6f8ff;
    padding: 1rem 1.2rem;
    border-radius: 0.8rem;
    border: 1px solid #d9e1ff;
}
.filter-panel h3, .filter-panel h4 {
    margin-top: 0.2rem;
    margin-bottom: 0.6rem;
}
.filter-panel label {
    font-weight: 500;
}
</style>
"""


def filter_panel_open():
    """Inject shared CSS and open the filter panel ``<div>``."""
    st.markdown(FILTER_PANEL_CSS, unsafe_allow_html=True)
    st.markdown("<div class='filter-panel'>", unsafe_allow_html=True)


def filter_panel_close():
    """Close the filter panel ``<div>``."""
    st.markdown("</div>", unsafe_allow_html=True)


@dataclass
class CompetScope:
    """Result of the competition-type filter."""
    choice: str   # "ALL" | "K1" | "SA" | "CHAMP"
    types: list   # Type_Compet codes kept
    data: pd.DataFrame

    @property
    def single_type(self):
        """The Type_Compet code when exactly one type is kept, else None."""
        return self.types[0] if len(self.types) == 1 else None


def compet_type_name(type_compet) -> str:
    """Readable (translated) name of a Type_Compet code."""
    return compet_type_label(type_compet, get_lang())


def type_compet_filter(data: pd.DataFrame, key_prefix: str, label: str = "Type de compétition") -> CompetScope:
    """Render the competition-type filter (Tous / K1 / SA / Championnats continentaux).

    When "Championnats continentaux" is chosen, a multiselect lets the user
    keep some of the championships present in the data (EKF, AKF, PKF, UFAK…).
    """
    present = [str(x) for x in data["Type_Compet"].dropna().unique()] if "Type_Compet" in data.columns else []
    champs = [c for c in CHAMPIONSHIP_TYPES if c in present]

    options = ["ALL", "K1", "SA"] + (["CHAMP"] if champs else [])
    names = {
        "ALL": t("Tous"),
        "K1": "Premier League (K1)",
        "SA": "Series A (SA)",
        "CHAMP": t("Championnats continentaux"),
    }
    choice = st.radio(t(label), options, format_func=lambda o: names[o], key=f"{key_prefix}_type_compet")

    if choice == "CHAMP":
        types = st.multiselect(
            t("Championnats"), champs, default=champs,
            format_func=compet_type_name, key=f"{key_prefix}_champs",
        ) or champs
    elif choice in ("K1", "SA"):
        types = [choice]
    else:
        types = present

    filtered = data[data["Type_Compet"].astype(str).isin(types)].copy() if choice != "ALL" else data.copy()
    return CompetScope(choice=choice, types=types, data=filtered)


def athlete_label_html(name: str) -> str:
    """Return a safe HTML snippet displaying an athlete's name."""
    safe_name = html_lib.escape(str(name))
    label = t("Athlète")
    return f"<p style='font-size:13px;font-style:italic;color:#555;'>{label} : {safe_name}</p>"


def highlight_victory_series(s):
    """Styler function: green for 'Oui'/'Yes', red for 'Non'/'No'."""
    styles = []
    for v in s:
        if v in ("Oui", "Yes"):
            styles.append("background-color: #d4edda;")
        elif v in ("Non", "No"):
            styles.append("background-color: #f8d7da;")
        else:
            styles.append("")
    return styles


def add_footer():
    """Render the global application footer."""
    footer = """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        right: 0;
        text-align: right;
        width: 100%;
        padding: 10px;
        font-size: 12px;
        color: grey;
        z-index: 9999;
    }
    .footer .source {
        color: lightgrey;
        opacity: 0.7;
    }
    .footer p {
        margin: 0;
    }
    </style>
    <div class="footer">
        <p class="source">Source : SportData</p>
        <p>&copy; Alexis Vincent</p>
    </div>
    """
    st.markdown(footer, unsafe_allow_html=True)
