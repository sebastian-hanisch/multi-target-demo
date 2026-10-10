"""Kennzahlen, Verdikte und Sweeps über die festen Instanzen (SWEEP_SEEDS)."""

from dataclasses import dataclass, replace

import numpy as np

import mtt_algorithm as A
import mtt_constants as C
import mtt_scenario as S


@dataclass(frozen=True)
class Settings:
    angle: float = C.DEFAULT_ANGLE          # Kreuzungswinkel in Grad
    dt: float = C.DEFAULT_DT                # Zeitversatz am Kreuzungspunkt in s (0 = beide zugleich)
    sigma: float = C.DEFAULT_SIGMA          # Messrauschen in m
    q: float = C.DEFAULT_Q                  # Fahrunruhe der Fahrzeuge in m/s^2 (Wahrheit UND Annahme des Filters)
    pd: float = C.DEFAULT_PD                # Entdeckungswahrscheinlichkeit je Fahrzeug und Scan
    clutter: float = C.DEFAULT_CLUTTER      # mittlere Fehlalarme je Scan
    seed: int = C.DEFAULT_SEED


@dataclass(frozen=True)
class TrackRun:
    pos: np.ndarray        # (T, 2, 2) Spurpositionen [Zeit, Spur, xy]
    wrong: int             # Korrekturen mit einer Messung, die nicht zum eigenen Fahrzeug gehört
    updates: int           # alle Korrekturen
    swapped: bool          # am Ende ist Spur 0 näher an Fahrzeug 1 (Identitäten vertauscht)


def identities_swapped(track_pos, truth_pos):
    """Vertauscht, wenn die Spuren am Ende insgesamt näher an dem jeweils anderen Fahrzeug liegen. Beide Argumente: (2, 2) [Objekt, xy]."""
    keep = np.linalg.norm(track_pos[0] - truth_pos[0]) + np.linalg.norm(track_pos[1] - truth_pos[1])
    cross = np.linalg.norm(track_pos[0] - truth_pos[1]) + np.linalg.norm(track_pos[1] - truth_pos[0])
    return bool(cross < keep)


def run_tracker(scene, v0_noise, method, sigma, q):
    """Zwei Spuren über die ganze Szene. method: 'nn' oder 'gnn'. Spuren starten mit bekannter Identität (Position der ersten Messung, ungefähre Geschwindigkeit)."""
    F, Q = A.cv_model_2d(C.DT, q)
    r = sigma ** 2
    xs, Ps = [], []
    for i in range(2):
        x0 = np.array([*(scene.truth[0, i] + sigma * 0.0), *(scene.vel[i] + C.V_INIT_STD * v0_noise[i])])
        xs.append(x0)
        Ps.append(np.diag([r, r, C.V_INIT_STD ** 2, C.V_INIT_STD ** 2]))
    T = len(scene.scans)
    pos = np.zeros((T, 2, 2))
    wrong = updates = 0
    assoc = A.associate_nn if method == "nn" else A.associate_gnn
    for t in range(T):
        if t > 0:
            for i in range(2):
                xs[i], Ps[i] = A.predict(xs[i], Ps[i], F, Q)
            Z, origin = scene.scans[t]
            if len(Z):
                Cm = A.cost_matrix(xs, Ps, Z, r)
                pick = assoc(Cm, C.GATE_PROB_CHI2)
                for i in range(2):
                    j = int(pick[i])
                    if j >= 0:
                        xs[i], Ps[i] = A.update(xs[i], Ps[i], Z[j], r)
                        updates += 1
                        wrong += int(origin[j] != i)
        for i in range(2):
            pos[t, i] = xs[i][:2]
    return TrackRun(pos, wrong, updates, identities_swapped(pos[-1], scene.truth[-1]))


@dataclass(frozen=True)
class Analysis:
    settings: Settings
    scene: S.Scene
    runs: dict                  # Methode -> TrackRun
    n_clutter: int              # Fehlalarme insgesamt
    n_missed: int               # verpasste Entdeckungen insgesamt
    rmse: dict                  # Methode -> RMSE der Spuren gegen das eigene Fahrzeug


def analyse(settings):
    scene, v0 = S.simulate(settings.seed, settings.angle, settings.dt, settings.sigma, settings.pd, settings.clutter, settings.q)
    runs = {m: run_tracker(scene, v0, m, settings.sigma, max(settings.q, C.Q_FLOOR)) for m in C.METHODS}
    n_clutter = int(sum((o == -1).sum() for _, o in scene.scans))
    n_det = int(sum((o >= 0).sum() for _, o in scene.scans))
    rmse = {m: float(np.sqrt(np.mean(np.sum((r.pos - scene.truth) ** 2, axis=2)))) for m, r in runs.items()}
    return Analysis(settings, scene, runs, n_clutter, 2 * C.T_STEPS - n_det, rmse)


# --- Sweeps ---------------------------------------------------------------------------------------------------------------------------------

def _summary(settings):
    res = [analyse(replace(settings, seed=s)) for s in C.SWEEP_SEEDS]
    out = {}
    for m in C.METHODS:
        w = sum(a.runs[m].wrong for a in res)
        u = sum(a.runs[m].updates for a in res)
        out[m] = {"swap": float(np.mean([a.runs[m].swapped for a in res])), "wrong": w / u if u else 0.0}
    return out


def sweep_angle(settings):
    return [{"angle": ang, **_summary(replace(settings, angle=ang))} for ang in C.ANGLE_SWEEP]


def sweep_clutter(settings):
    return [{"clutter": lam, **_summary(replace(settings, clutter=lam))} for lam in C.CLUTTER_SWEEP]


def sweep_detection(settings):
    return [{"pd": pd, **_summary(replace(settings, pd=pd))} for pd in C.PD_SWEEP]


def sweep_offset(settings):
    return [{"dt": d, **_summary(replace(settings, dt=d))} for d in C.OFFSET_SWEEP]


def sweep_unrest(settings):
    return [{"q": q, **_summary(replace(settings, q=q))} for q in C.Q_SWEEP]
