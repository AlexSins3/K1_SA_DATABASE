"""Tests for utils/data_helpers.py"""

import numpy as np
import pandas as pd
import pytest

from utils.data_helpers import (
    victoire_to_bool,
    victoire_to_str,
    victoire_to_int,
    normalize_victoire_category,
    build_compet_label,
    safe_mode,
    get_numeric_and_categorical_columns,
)


class TestVictoireToBool:
    def test_true_values(self):
        for v in [True, 1, "true", "True", "1", "oui", "Vrai", "yes", "win", "gagné"]:
            assert victoire_to_bool(v) is True, f"Expected True for {v!r}"

    def test_false_values(self):
        for v in [False, 0, "false", "False", "0", "non", "faux", "no", "perdu"]:
            assert victoire_to_bool(v) is False, f"Expected False for {v!r}"

    def test_none_values(self):
        assert victoire_to_bool(None) is None
        assert victoire_to_bool(np.nan) is None
        assert victoire_to_bool(pd.NA) is None


class TestVictoireToStr:
    def test_oui(self):
        assert victoire_to_str(True) == "Oui"
        assert victoire_to_str(1) == "Oui"

    def test_non(self):
        assert victoire_to_str(False) == "Non"
        assert victoire_to_str(0) == "Non"

    def test_inconnu(self):
        assert victoire_to_str(None) == "Inconnu"
        assert victoire_to_str(np.nan) == "Inconnu"


class TestVictoireToInt:
    def test_int_values(self):
        assert victoire_to_int(True) == 1
        assert victoire_to_int(False) == 0
        assert victoire_to_int("oui") == 1
        assert victoire_to_int("non") == 0
        assert victoire_to_int(None) is None


class TestNormalizeVictoireCategory:
    def test_returns_strings(self):
        assert normalize_victoire_category(True) == "True"
        assert normalize_victoire_category(False) == "False"
        assert np.isnan(normalize_victoire_category(None))


class TestBuildCompetLabel:
    def test_with_year(self):
        assert build_compet_label("Paris", 2024) == "Paris_2024"

    def test_without_year(self):
        assert build_compet_label("Paris") == "Paris"
        assert build_compet_label("Paris", None) == "Paris"
        assert build_compet_label("Paris", np.nan) == "Paris"


class TestSafeMode:
    def test_normal(self):
        s = pd.Series(["a", "a", "b"])
        assert safe_mode(s) == "a"

    def test_empty(self):
        s = pd.Series([], dtype="object")
        assert safe_mode(s, default="X") == "X"

    def test_all_nan(self):
        s = pd.Series([np.nan, np.nan])
        assert safe_mode(s, default="X") == "X"


class TestGetNumericAndCategorical:
    def test_split(self):
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"], "c": [1.0, 2.0]})
        num, cat = get_numeric_and_categorical_columns(df)
        assert "a" in num
        assert "c" in num
        assert "b" in cat
