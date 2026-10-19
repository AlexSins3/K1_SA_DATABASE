"""Tests for competition types, chronology, tour labels and flag classification."""

from constants.competitions import (
    get_compet_chrono_rank,
    is_championship,
    max_flags,
    panel_group,
)
from constants.tours import map_tour_label, tour_level
from utils.data_helpers import classify_match_flags
from utils.display import fmt_tour


class TestCompetTypes:
    def test_judges(self):
        assert max_flags("K1") == 7
        assert max_flags("SA") == 5
        assert max_flags("EKF") == 7
        assert max_flags("AKF") == 7
        assert max_flags("PKF") == 5
        assert max_flags("UFAK") == 7

    def test_panel_groups(self):
        assert {panel_group(c) for c in ("K1", "EKF", "AKF", "UFAK")} == {"J7"}
        assert {panel_group(c) for c in ("SA", "PKF")} == {"J5"}

    def test_is_championship(self):
        assert is_championship("EKF") and is_championship("UFAK")
        assert not is_championship("K1") and not is_championship("SA")


class TestChronology:
    def test_2026_order(self):
        order = ["K1_Leshan", "SA_ACoruna", "EKF_Frankfurt", "PKF_Rio",
                 "K1_Rabat", "AKF_Bali", "UFAK_Algiers", "SA_Salzbourg"]
        ranks = [get_compet_chrono_rank(c, 2026) for c in order]
        assert ranks == sorted(ranks)
        assert len(set(ranks)) == len(ranks)

    def test_same_city_different_year(self):
        assert get_compet_chrono_rank("K1 Rabat", 2025) < get_compet_chrono_rank("K1 Rabat", 2026)

    def test_unknown_competition_falls_back_to_year(self):
        assert get_compet_chrono_rank("EKF_Unknown", 2027) > get_compet_chrono_rank("SA_Salzbourg", 2026)


class TestClassifyMatchFlags:
    def test_seven_judges(self):
        assert classify_match_flags(4, 3) == "Serré (1 drapeau)"
        assert classify_match_flags(5, 2) == "Net (2-3 drapeaux)"
        assert classify_match_flags(6, 1) == "Net (2-3 drapeaux)"
        assert classify_match_flags(7, 0) == "Unanime"

    def test_five_judges(self):
        assert classify_match_flags(3, 2) == "Serré (1 drapeau)"
        assert classify_match_flags(4, 1) == "Net (2-3 drapeaux)"
        assert classify_match_flags(5, 0) == "Unanime"


class TestTourLabels:
    def test_sm_is_semi_final(self):
        for c in ("EKF", "AKF", "PKF", "UFAK"):
            assert map_tour_label("SM", c) == "Demi finale"

    def test_format_specific_stage(self):
        assert map_tour_label("PW1", "EKF") == "Quart de finale"
        assert map_tour_label("T3", "AKF") == "Quart de finale"
        assert map_tour_label("PW1", "SA") == "Finale de poule"

    def test_levels_increase_to_final(self):
        for c in ("EKF", "AKF", "UFAK"):
            assert tour_level(map_tour_label("T2", c), c) < tour_level(map_tour_label("SM", c), c)
            assert tour_level(map_tour_label("SM", c), c) < tour_level(map_tour_label("Final", c), c)

    def test_fmt_tour(self):
        assert fmt_tour("T1", "SA") == "1er tour Series A"
        assert fmt_tour("PW1", "SA") == "8ème de finale Series A"
        assert fmt_tour("PW1", "EKF") == "Quart de finale EKF"
        assert fmt_tour("SM", "AKF") == "Demi-finale AKF"
        assert fmt_tour("SM") == "Demi-finale"
        assert fmt_tour("R1") == "Quart de finale K1"
