"""Szenario: zwei Fahrzeuge kreuzen sich; ein Sensor liefert anonyme Positionsmessungen mit Aussetzern und Fehlalarmen."""

from dataclasses import dataclass

import numpy as np

import mtt_constants as C

MAX_CLUTTER = 12          # Obergrenze der vorab gezogenen Fehlalarm-Kandidaten je Scan


@dataclass(frozen=True)
class Scene:
    t: np.ndarray               # (T,)
    truth: np.ndarray           # (T, 2, 2) wahre Positionen [Zeit, Fahrzeug, xy]
    vel: np.ndarray             # (2, 2) wahre Geschwindigkeiten [Fahrzeug, xy]
    scans: list                 # Länge T: (Z (m, 2) Messungen, origin (m,) -1 = Fehlalarm, sonst Fahrzeugnummer)
    sigma: float
    min_separation: float       # kleinster Abstand der beiden wahren Bahnen


def _poisson_from_uniform(u, lam):
    """Poisson-Zahl aus EINER gleichverteilten Zahl (Inversion): monoton in lam, damit ein Regler die Spur stetig ändert."""
    if lam <= 0:
        return 0
    k, p = 0, np.exp(-lam)
    cum = p
    while u > cum and k < MAX_CLUTTER:
        k += 1
        p *= lam / k
        cum += p
    return k


def simulate(seed, angle_deg, dt_offset, sigma, pd, clutter, q=0.0):
    """Wahre Bahnen, Messungen je Sekunde. Alle Zufallszahlen kommen in fester Reihenfolge aus einem Generator; die Regler schwellen oder skalieren sie nur."""
    rng = np.random.default_rng(seed)
    T = C.T_STEPS
    u_det = rng.random((T, 2))
    noise = rng.standard_normal((T, 2, 2))
    u_cnt = rng.random(T)
    cl_pts = rng.uniform(-C.CLUTTER_BOX, C.CLUTTER_BOX, (T, MAX_CLUTTER, 2))
    perm_keys = rng.random((T, 2 + MAX_CLUTTER))
    v0_noise = rng.standard_normal((2, 2))
    n_proc = rng.standard_normal((T, 2, 4))
    half = np.deg2rad(angle_deg) / 2.0
    ua = np.array([np.cos(half), np.sin(half)])
    ub = np.array([np.cos(half), -np.sin(half)])
    t = np.arange(T) * C.DT
    mid = T / 2.0
    truth = np.zeros((T, 2, 2))
    vel_end = np.zeros((2, 2))
    if q > 0:
        from mtt_algorithm import cv_model_2d
        F, Q = cv_model_2d(C.DT, q)
        L = np.linalg.cholesky(Q + 1e-12 * np.eye(4))
    for i, (u, off) in enumerate(((ua, 0.0), (ub, dt_offset))):
        x = np.array([*(C.SPEED * u * (-mid - off)), *(C.SPEED * u)])
        for k in range(T):
            truth[k, i] = x[:2]
            if q > 0:
                x = F @ x + L @ n_proc[k, i]
            else:
                x = x + np.array([C.DT * x[2], C.DT * x[3], 0.0, 0.0])
    scans = []
    for k in range(T):
        pts, org = [], []
        for i in range(2):
            if u_det[k, i] < pd:
                pts.append(truth[k, i] + sigma * noise[k, i])
                org.append(i)
        for c in range(_poisson_from_uniform(u_cnt[k], clutter)):
            pts.append(cl_pts[k, c])
            org.append(-1)
        if pts:
            order = np.argsort(perm_keys[k, :len(pts)])
            Z = np.array(pts)[order]
            origin = np.array(org)[order]
        else:
            Z, origin = np.zeros((0, 2)), np.zeros(0, int)
        scans.append((Z, origin))
    vel = np.array([C.SPEED * ua, C.SPEED * ub])
    sep = float(np.min(np.linalg.norm(truth[:, 0] - truth[:, 1], axis=1)))
    return Scene(t, truth, vel, scans, sigma, sep), v0_noise
