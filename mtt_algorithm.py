"""Mehrzielverfolgung, numpy-only: je Fahrzeug ein Kalman-Filter, dazu Datenzuordnung (nächste Messung / globale Zuordnung per Ungarischer Methode).

Zustand je Fahrzeug x = (x, y, vx, vy), konstante Geschwindigkeit mit zufälliger Beschleunigung; gemessen wird die Position, anonym (ohne Identität).
Jede Funktion ist eine Einheit mit expliziten Ein- und Ausgaben.
"""

import numpy as np

H_POS = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])


def cv_model_2d(dt, q):
    """F und Q des Konstante-Geschwindigkeit-Modells in der Ebene, Reihenfolge (x, y, vx, vy)."""
    F = np.eye(4)
    F[0, 2] = F[1, 3] = dt
    Q = np.zeros((4, 4))
    for a in (0, 1):
        Q[a, a] = q ** 2 * dt ** 3 / 3.0
        Q[a, a + 2] = Q[a + 2, a] = q ** 2 * dt ** 2 / 2.0
        Q[a + 2, a + 2] = q ** 2 * dt
    return F, Q


def predict(x, P, F, Q):
    return F @ x, F @ P @ F.T + Q


def update(x, P, z, r):
    """Kalman-Korrektur mit einer Positionsmessung z (Varianz r je Achse), Joseph-Form."""
    R = np.eye(2) * r
    S = H_POS @ P @ H_POS.T + R
    K = P @ H_POS.T @ np.linalg.inv(S)
    I_KH = np.eye(4) - K @ H_POS
    return x + K @ (z - H_POS @ x), I_KH @ P @ I_KH.T + K @ R @ K.T


def cost_matrix(tracks_x, tracks_P, Z, r):
    """Kosten je Spur und Messung: normiertes Innovationsquadrat nu' S^-1 nu (Mahalanobis). Form (Spuren, Messungen)."""
    C = np.zeros((len(tracks_x), len(Z)))
    for i, (x, P) in enumerate(zip(tracks_x, tracks_P)):
        S = H_POS @ P @ H_POS.T + np.eye(2) * r
        Sinv = np.linalg.inv(S)
        for j, z in enumerate(Z):
            nu = z - H_POS @ x
            C[i, j] = float(nu @ Sinv @ nu)
    return C


def hungarian(cost):
    """Ungarische Methode (Kuhn/Munkres, mit Potentialen): optimale Zuordnung jeder Zeile (Spur) zu einer eigenen Spalte (Messung), n Zeilen <= m Spalten.

    Gibt für jede Zeile den Spaltenindex zurück, Summe der Kosten minimal.
    """
    cost = np.asarray(cost, float)
    n, m = cost.shape
    if n > m:
        raise ValueError("mehr Zeilen als Spalten")
    INF = float("inf")
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, int)       # p[j]: Zeile, die Spalte j zugeordnet ist (1-basiert, 0 = frei)
    way = np.zeros(m + 1, int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, INF)
        used = np.zeros(m + 1, bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta, j1 = INF, 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assign = np.zeros(n, int)
    for j in range(1, m + 1):
        if p[j] != 0:
            assign[p[j] - 1] = j - 1
    return assign


def associate_nn(C, gate):
    """Nächste Messung je Spur, unabhängig voneinander (zwei Spuren dürfen dieselbe Messung nehmen). -1 = keine Messung im Tor."""
    out = np.full(C.shape[0], -1, int)
    if C.shape[1] == 0:
        return out
    for i in range(C.shape[0]):
        j = int(C[i].argmin())
        if C[i, j] <= gate:
            out[i] = j
    return out


def associate_gnn(C, gate):
    """Globale Zuordnung: jede Messung höchstens einer Spur, Gesamtkosten minimal; Paare außerhalb des Tors bleiben ungepaart (-1).

    Aufgefüllt mit einer „keine Messung“-Spalte je Spur zu den Kosten `gate`: eine Spur nimmt eine Messung nur, wenn sie billiger ist als das Tor.
    """
    n, m = C.shape
    out = np.full(n, -1, int)
    if n == 0:
        return out
    padded = np.concatenate([np.where(C <= gate, C, gate + 1e6) if m else np.zeros((n, 0)), np.full((n, n), gate)], axis=1)
    cols = hungarian(padded)
    for i, j in enumerate(cols):
        if j < m and C[i, j] <= gate:
            out[i] = int(j)
    return out
