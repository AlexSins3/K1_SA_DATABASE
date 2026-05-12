"""Tests for utils/stats.py"""

import numpy as np
import pandas as pd
import pytest

from utils.stats import (
    fmt_p,
    normality_test_auto,
    oneway_test_auto,
    chi2_or_fisher_auto,
    correlation_auto,
)


class TestFmtP:
    def test_small_p(self):
        assert fmt_p(0.0001) == "< 0.001"

    def test_normal_p(self):
        assert fmt_p(0.05) == "0.050"

    def test_large_p(self):
        assert fmt_p(0.95) == "0.950"


class TestNormality:
    def test_insufficient_data(self):
        res = normality_test_auto(pd.Series([1.0, 2.0]))
        assert np.isnan(res["stat"])
        assert "insuffisant" in res["comment"].lower()

    def test_small_sample_shapiro(self):
        np.random.seed(42)
        sample = pd.Series(np.random.normal(0, 1, 10))
        res = normality_test_auto(sample)
        assert res["test_name"] == "Shapiro-Wilk"
        assert not np.isnan(res["p"])

    def test_large_sample_dagostino(self):
        np.random.seed(42)
        sample = pd.Series(np.random.normal(0, 1, 100))
        res = normality_test_auto(sample)
        assert res["test_name"] == "D'Agostino K²"
        assert not np.isnan(res["p"])


class TestOneway:
    def test_insufficient_groups(self):
        df = pd.DataFrame({"y": [1, 2], "x": ["a", "a"]})
        res = oneway_test_auto(df, "y", "x")
        assert np.isnan(res["stat"])

    def test_two_groups(self):
        np.random.seed(42)
        df = pd.DataFrame({
            "y": np.concatenate([np.random.normal(0, 1, 30), np.random.normal(3, 1, 30)]),
            "x": ["a"] * 30 + ["b"] * 30,
        })
        res = oneway_test_auto(df, "y", "x")
        assert res["p"] < 0.05


class TestChi2:
    def test_basic(self):
        df = pd.DataFrame({
            "x": ["a", "a", "b", "b"] * 10,
            "y": ["c", "d", "c", "d"] * 10,
        })
        res = chi2_or_fisher_auto(df, "x", "y")
        assert "test_name" in res
        assert res["p"] is not None


class TestCorrelation:
    def test_insufficient(self):
        res = correlation_auto(pd.Series([1.0, 2.0]), pd.Series([3.0, np.nan]))
        assert len(res["tests"]) == 0

    def test_correlated(self):
        np.random.seed(42)
        x = pd.Series(np.arange(50, dtype=float))
        y = x * 2 + np.random.normal(0, 1, 50)
        res = correlation_auto(x, y)
        assert len(res["tests"]) > 0
        assert res["tests"][0]["p"] < 0.05
