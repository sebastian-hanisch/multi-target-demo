"""Orakel-Test multi-target-demo: Zuordnung und Verfolgung gegen ANDERE Rechenwege.

1. Ungarische Methode gegen scipy.optimize.linear_sum_assignment und gegen Aufzählung aller Zuordnungen,
2. globale Zuordnung mit Tor und der Möglichkeit „keine Messung“ gegen Aufzählung aller zulässigen Zuordnungen,
3. mit richtiger Zuordnung ist das Ergebnis des Trackers gleich dem Kalman-Filter mit bekannter Identität,
4. Tor-Rate auf echten Messungen gleich 1 - Quantil, Poisson-Mittel der Fehlalarme,
5. die angezeigten Kennzahlen gegen eine unabhängige Neuberechnung.
"""

import itertools

import numpy as np
import pytest

import mtt_algorithm as A
import mtt_constants as C
import mtt_evaluation as ev
import mtt_scenario as S


def _brute_force_cost(cost):
    n, m = cost.shape
    return min(sum(cost[i, p[i]] for i in range(n)) for p in itertools.permutations(range(m), n))


def test_hungarian_matches_scipy_and_brute_force():
    opt = pytest.importorskip("scipy.optimize")
    rng = np.random.default_rng(0)
    for _ in range(150):
        n = int(rng.integers(1, 5))
        m = int(rng.integers(n, 7))
        cost = rng.uniform(0, 10, (n, m))
        if rng.random() < 0.3:
            cost = np.round(cost)                                       # Gleichstände
        cols = A.hungarian(cost)
        assert len(set(cols.tolist())) == n                              # jede Messung höchstens einmal
        total = cost[np.arange(n), cols].sum()
        r, c = opt.linear_sum_assignment(cost)
        assert total == pytest.approx(cost[r, c].sum())
        assert total == pytest.approx(_brute_force_cost(cost))


def _gated_brute_force(C_, gate):
    """Kleinste Gesamtkosten, wenn jede Spur höchstens eine Messung bekommt (jede Messung höchstens einmal) oder keine zu den Kosten gate; Paare über dem Tor verboten."""
    n, m = C_.shape
    best = np.inf
    for choice in itertools.product(range(-1, m), repeat=n):
        used = [j for j in choice if j >= 0]
        if len(used) != len(set(used)):
            continue
        if any(j >= 0 and C_[i, j] > gate for i, j in enumerate(choice)):
            continue
        total = sum(gate if j < 0 else C_[i, j] for i, j in enumerate(choice))
        best = min(best, total)
    return best


def test_gnn_with_gate_and_unassigned_option_is_optimal():
    rng = np.random.default_rng(1)
    for _ in range(200):
        n = int(rng.integers(1, 4))
        m = int(rng.integers(0, 5))
        C_ = rng.uniform(0, 14, (n, m))
        gate = 9.2
        pick = A.associate_gnn(C_, gate)
        total = sum(gate if j < 0 else C_[i, j] for i, j in enumerate(pick))
        assert total == pytest.approx(_gated_brute_force(C_, gate))
        used = [j for j in pick if j >= 0]
        assert len(used) == len(set(used))


def _oracle_tracker(scene, v0, sigma, q):
    """Kalman-Filter mit BEKANNTER Zuordnung (Wahrheit): jede Spur bekommt die Messung ihres eigenen Fahrzeugs, wenn es entdeckt wurde."""
    F, Q = A.cv_model_2d(C.DT, q)
    r = sigma ** 2
    xs = [np.array([*scene.truth[0, i], *(scene.vel[i] + C.V_INIT_STD * v0[i])]) for i in range(2)]
    Ps = [np.diag([r, r, C.V_INIT_STD ** 2, C.V_INIT_STD ** 2]) for _ in range(2)]
    pos = np.zeros((C.T_STEPS, 2, 2))
    for t in range(C.T_STEPS):
        if t > 0:
            Z, origin = scene.scans[t]
            for i in range(2):
                xs[i], Ps[i] = A.predict(xs[i], Ps[i], F, Q)
                hit = np.flatnonzero(origin == i)
                if len(hit):
                    xs[i], Ps[i] = A.update(xs[i], Ps[i], Z[hit[0]], r)
        for i in range(2):
            pos[t, i] = xs[i][:2]
    return pos


def test_tracker_equals_known_association_filter_when_every_association_is_right():
    """Mit Zeitversatz und scharfer Messung sind alle Zuordnungen richtig. Wo das Tor eine echte Messung ablehnt (etwa 1 %), fehlt dem Tracker eine Korrektur;
    nur Läufe, in denen alle Korrekturen stattfanden, sind mit dem Filter bekannter Identität vergleichbar."""
    compared = 0
    for seed in range(20):
        s = ev.Settings(angle=90.0, dt=4.0, sigma=3.0, pd=1.0, clutter=0.0, q=0.1, seed=seed)
        a = ev.analyse(s)
        scene, v0 = S.simulate(seed, s.angle, s.dt, s.sigma, s.pd, s.clutter, s.q)
        ref = _oracle_tracker(scene, v0, s.sigma, max(s.q, C.Q_FLOOR))
        for m in C.METHODS:
            assert a.runs[m].wrong == 0, (seed, m)
            if a.runs[m].updates == 2 * (C.T_STEPS - 1):
                assert a.runs[m].pos == pytest.approx(ref, abs=1e-9), (seed, m)
                compared += 1
    assert compared >= 6


def test_gate_rejects_about_one_percent_of_true_measurements():
    exceed = total = 0
    for seed in range(60):
        scene, v0 = S.simulate(seed, 90.0, 0.0, 5.0, 1.0, 0.0, 0.1)
        F, Q = A.cv_model_2d(C.DT, 0.1)
        r = 25.0
        xs = [np.array([*scene.truth[0, i], *(scene.vel[i] + C.V_INIT_STD * v0[i])]) for i in range(2)]
        Ps = [np.diag([r, r, C.V_INIT_STD ** 2, C.V_INIT_STD ** 2]) for _ in range(2)]
        for t in range(1, C.T_STEPS):
            Z, origin = scene.scans[t]
            for i in range(2):
                xs[i], Ps[i] = A.predict(xs[i], Ps[i], F, Q)
            Cm = A.cost_matrix(xs, Ps, Z, r)
            for i in range(2):
                j = int(np.flatnonzero(origin == i)[0])
                exceed += Cm[i, j] > C.GATE_PROB_CHI2
                total += 1
                xs[i], Ps[i] = A.update(xs[i], Ps[i], Z[j], r)
    assert 0.003 < exceed / total < 0.02


def test_clutter_count_has_the_requested_mean():
    for lam in (0.5, 1.0, 2.0):
        counts = []
        for seed in range(80):
            scene, _ = S.simulate(seed, 20.0, 0.0, 10.0, 1.0, lam, 0.0)
            counts.extend(int((o < 0).sum()) for _, o in scene.scans)
        assert np.mean(counts) == pytest.approx(lam, rel=0.08)


def test_displayed_numbers_match_independent_recomputation():
    for seed in (3, 11, 21):
        s = ev.Settings(seed=seed, clutter=1.0)
        a = ev.analyse(s)
        scene, v0 = S.simulate(seed, s.angle, s.dt, s.sigma, s.pd, s.clutter, s.q)
        assert a.n_clutter == int(sum((o < 0).sum() for _, o in scene.scans))
        assert a.n_missed == 2 * C.T_STEPS - int(sum((o >= 0).sum() for _, o in scene.scans))
        for m in C.METHODS:
            r = a.runs[m]
            d_keep = np.linalg.norm(r.pos[-1, 0] - scene.truth[-1, 0]) + np.linalg.norm(r.pos[-1, 1] - scene.truth[-1, 1])
            d_swap = np.linalg.norm(r.pos[-1, 0] - scene.truth[-1, 1]) + np.linalg.norm(r.pos[-1, 1] - scene.truth[-1, 0])
            assert r.swapped == (d_swap < d_keep)
            assert 0 <= r.wrong <= r.updates <= 2 * (C.T_STEPS - 1)
        assert a.rmse["nn"] == pytest.approx(np.sqrt(np.mean(np.sum((a.runs["nn"].pos - scene.truth) ** 2, axis=2))))
