"""Jede Zahl aus dem README, nachgerechnet über dieselben Auswertungsfunktionen wie die App (100 feste Begegnungen).

Plattformrobust: Bänder statt exakter Werte (CI installiert neueste numpy/scipy)."""

from dataclasses import replace

import pytest

import mtt_constants as C
import mtt_evaluation as ev


@pytest.fixture(scope="module")
def angle_rows():
    return {r["angle"]: r for r in ev.sweep_angle(ev.Settings())}


@pytest.fixture(scope="module")
def offset_rows():
    return {r["dt"]: r for r in ev.sweep_offset(ev.Settings())}


@pytest.fixture(scope="module")
def unrest_rows():
    return {r["q"]: r for r in ev.sweep_unrest(ev.Settings())}


@pytest.fixture(scope="module")
def clutter_rows():
    return {r["clutter"]: r for r in ev.sweep_clutter(ev.Settings())}


def test_global_association_makes_fewer_wrong_associations_at_every_angle(angle_rows):
    for ang, r in angle_rows.items():
        assert r["gnn"]["wrong"] < r["nn"]["wrong"], ang
    r10, r45 = angle_rows[10.0], angle_rows[45.0]
    assert r10["nn"]["wrong"] == pytest.approx(0.38, abs=0.05) and r10["gnn"]["wrong"] == pytest.approx(0.26, abs=0.05)
    assert r45["nn"]["wrong"] == pytest.approx(0.08, abs=0.03) and r45["gnn"]["wrong"] == pytest.approx(0.04, abs=0.02)


def test_identity_swaps_vanish_at_steeper_angles(angle_rows):
    assert angle_rows[10.0]["gnn"]["swap"] == pytest.approx(0.30, abs=0.12)
    assert angle_rows[10.0]["nn"]["swap"] == pytest.approx(0.13, abs=0.09)
    for ang in (45.0, 60.0, 90.0):
        assert angle_rows[ang]["nn"]["swap"] <= 0.03 and angle_rows[ang]["gnn"]["swap"] <= 0.03


def test_time_offset_reverses_the_order_of_the_two_rules(offset_rows):
    simultaneous, shifted = offset_rows[0.0], offset_rows[4.0]
    assert simultaneous["gnn"]["swap"] > simultaneous["nn"]["swap"] + 0.05
    assert shifted["nn"]["swap"] > shifted["gnn"]["swap"] + 0.10
    assert shifted["nn"]["swap"] == pytest.approx(0.28, abs=0.10) and shifted["gnn"]["swap"] == pytest.approx(0.06, abs=0.06)


def test_unrest_makes_the_simple_rule_swap_more_than_the_global_one(unrest_rows):
    q0, q3 = unrest_rows[0.0], unrest_rows[0.3]
    assert q0["nn"]["swap"] <= 0.03 and q0["gnn"]["swap"] == pytest.approx(0.23, abs=0.10)
    assert q3["nn"]["swap"] == pytest.approx(0.49, abs=0.12) and q3["gnn"]["swap"] == pytest.approx(0.30, abs=0.10)
    assert q3["nn"]["swap"] > q3["gnn"]["swap"]


def test_clutter_hardly_changes_anything_in_this_setting(clutter_rows):
    lo, hi = clutter_rows[0.0], clutter_rows[3.0]
    for m in C.METHODS:
        assert abs(hi[m]["wrong"] - lo[m]["wrong"]) < 0.06
        assert abs(hi[m]["swap"] - lo[m]["swap"]) < 0.10


def test_sweeps_use_the_fixed_instances():
    assert len(C.SWEEP_SEEDS) == 100 and C.SWEEP_SEEDS[0] == 100000
    assert ev.Settings().angle == 10.0 and replace(ev.Settings(), dt=4.0).dt == 4.0
