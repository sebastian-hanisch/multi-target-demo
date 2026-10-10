"""Einheiten des Zuordnungs- und Filterkerns gegen Handrechnung und Randfälle."""

import numpy as np
import pytest

import mtt_algorithm as A
import mtt_constants as C
import mtt_evaluation as ev
import mtt_scenario as S


def test_hungarian_hand_examples():
    assert A.hungarian(np.array([[1.0, 2.0], [2.0, 1.0]])).tolist() == [0, 1]
    assert A.hungarian(np.array([[2.0, 1.0], [1.0, 2.0]])).tolist() == [1, 0]
    assert A.hungarian(np.array([[5.0, 1.0, 9.0], [5.0, 9.0, 2.0]])).tolist() == [1, 2]       # rechteckig: 2 Spuren, 3 Messungen


def test_hungarian_rejects_more_rows_than_columns():
    with pytest.raises(ValueError):
        A.hungarian(np.zeros((3, 2)))


def test_nn_lets_two_tracks_share_a_measurement_gnn_does_not(cost_example):
    assert A.associate_nn(cost_example, 9.0).tolist() == [0, 0]
    assert A.associate_gnn(cost_example, 9.0).tolist() == [0, 1]


def test_gate_excludes_pairs_beyond_the_threshold(cost_example):
    assert A.associate_nn(cost_example, 0.5).tolist() == [-1, -1]
    assert A.associate_gnn(cost_example, 0.5).tolist() == [-1, -1]
    assert A.associate_nn(cost_example, 4.5).tolist() == [0, 0]
    # Messung 1 kostet Spur 1 nur 4: sie liegt innerhalb des Tors 4,5, aber Spur 0 hat Messung 0 näher: global bekommt Spur 1 die Messung 1
    assert A.associate_gnn(cost_example, 4.5).tolist() == [0, 1]
    assert A.associate_gnn(cost_example, 3.0).tolist() == [0, -1]


def test_no_measurements_means_no_association():
    empty = np.zeros((2, 0))
    assert A.associate_nn(empty, 9.0).tolist() == [-1, -1]
    assert A.associate_gnn(empty, 9.0).tolist() == [-1, -1]


def test_more_tracks_than_measurements_leaves_the_worse_one_unassigned():
    C_ = np.array([[1.0], [2.0]])
    assert A.associate_gnn(C_, 9.0).tolist() == [0, -1]


def test_cost_matrix_hand_example():
    x = np.zeros(4)
    P = np.eye(4)
    Z = np.array([[2.0, 0.0], [0.0, 3.0]])
    C_ = A.cost_matrix([x], [P], Z, 1.0)
    # S = P_pos + r I = 2 I: Kosten = |nu|^2 / 2 = 2 und 4,5
    assert C_ == pytest.approx(np.array([[2.0, 4.5]]))


def test_update_hand_example():
    x, P = A.update(np.zeros(4), np.eye(4), np.array([2.0, 0.0]), 1.0)
    assert x == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert P[0, 0] == pytest.approx(0.5) and P[2, 2] == pytest.approx(1.0)


def test_swap_definition_hand_example():
    truth = np.array([[0.0, 0.0], [10.0, 0.0]])
    assert ev.identities_swapped(np.array([[0.5, 0.0], [9.5, 0.0]]), truth) is False
    assert ev.identities_swapped(np.array([[9.5, 0.0], [0.5, 0.0]]), truth) is True


def test_simulation_is_reproducible_and_sigma_scales_the_same_noise():
    a, _ = S.simulate(7, 20.0, 0.0, 10.0, 0.9, 1.0, 0.1)
    b, _ = S.simulate(7, 20.0, 0.0, 10.0, 0.9, 1.0, 0.1)
    assert np.array_equal(a.truth, b.truth) and all(np.array_equal(x[0], y[0]) for x, y in zip(a.scans, b.scans))
    c, _ = S.simulate(7, 20.0, 0.0, 20.0, 0.9, 1.0, 0.1)
    assert np.array_equal(a.truth, c.truth)
    for (za, oa), (zc, oc) in zip(a.scans, c.scans):
        assert np.array_equal(oa, oc)                                # dieselben Messungen in derselben Reihenfolge


def test_detection_and_clutter_are_nested_across_regler_values():
    lo, _ = S.simulate(3, 20.0, 0.0, 10.0, 0.5, 0.5, 0.1)
    hi, _ = S.simulate(3, 20.0, 0.0, 10.0, 0.9, 2.0, 0.1)
    n_lo = sum(len(z) for z, _ in lo.scans)
    n_hi = sum(len(z) for z, _ in hi.scans)
    assert n_hi >= n_lo
    det_lo = sum(int((o >= 0).sum()) for _, o in lo.scans)
    det_hi = sum(int((o >= 0).sum()) for _, o in hi.scans)
    assert det_hi >= det_lo
    assert all(o.size == 0 or (o >= -1).all() for _, o in hi.scans)


def test_truth_crosses_at_the_middle_and_time_offset_separates_them():
    sc, _ = S.simulate(1, 30.0, 0.0, 10.0, 1.0, 0.0, 0.0)
    assert sc.min_separation == pytest.approx(0.0, abs=1e-9)
    assert np.linalg.norm(sc.truth[C.T_STEPS // 2, 0]) == pytest.approx(0.0, abs=1e-9)
    off, _ = S.simulate(1, 30.0, 4.0, 10.0, 1.0, 0.0, 0.0)
    assert off.min_separation > 5.0


def test_poisson_from_uniform_is_monotone_in_the_rate_and_zero_for_zero_rate():
    for u in (0.1, 0.5, 0.9, 0.99):
        counts = [S._poisson_from_uniform(u, lam) for lam in (0.0, 0.5, 1.0, 2.0, 3.0)]
        assert counts == sorted(counts) and counts[0] == 0


def test_constants_are_consistent():
    assert C.DEFAULT_ANGLE in C.ANGLE_SWEEP and C.DEFAULT_PD in C.PD_SWEEP and 0.0 in C.OFFSET_SWEEP and 0.0 in C.Q_SWEEP
    assert C.T_STEPS % 2 == 0 and C.DEFAULT_DT == 0.0
