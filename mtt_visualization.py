"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go

import mtt_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=340):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=70), legend=dict(orientation="h", y=-0.3), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_scene_map(a, method):
    """Draufsicht: wahre Bahnen, anonyme Messungen (Fehlalarme rot), die beiden Spuren der gewählten Zuordnung (Farbe = Identität zum Start)."""
    sc = a.scene
    run = a.runs[method]
    fig = go.Figure()
    Z = np.vstack([z for z, _ in sc.scans if len(z)]) if any(len(z) for z, _ in sc.scans) else np.zeros((0, 2))
    org = np.concatenate([o for z, o in sc.scans if len(z)]) if len(Z) else np.zeros(0, int)
    fig.add_trace(go.Scatter(x=Z[org >= 0, 0], y=Z[org >= 0, 1], mode="markers", marker=dict(color=C.COLOR_RAW, size=4), name="Messung (echt)", hoverinfo="skip"))
    if (org < 0).any():
        fig.add_trace(go.Scatter(x=Z[org < 0, 0], y=Z[org < 0, 1], mode="markers", marker=dict(color=C.COLOR_CLUTTER, size=6, symbol="x"), name="Fehlalarm",
                                 hoverinfo="skip"))
    for i in range(2):
        fig.add_trace(go.Scatter(x=sc.truth[:, i, 0], y=sc.truth[:, i, 1], mode="lines", line=dict(color=C.COLOR_TRUTH[i], width=5, dash="dot"), name=f"Fahrzeug {i + 1} (wahr)"))
    for i in range(2):
        fig.add_trace(go.Scatter(x=run.pos[:, i, 0], y=run.pos[:, i, 1], mode="lines", line=dict(color=C.COLOR_TRACK[i], width=2), name=f"Spur {i + 1}"))
        fig.add_trace(go.Scatter(x=[run.pos[-1, i, 0]], y=[run.pos[-1, i, 1]], mode="markers", marker=dict(color=C.COLOR_TRACK[i], size=11, symbol="circle"),
                                 showlegend=False, hoverinfo="skip"))
    lim = float(np.abs(sc.truth).max()) + 30
    fig.add_trace(go.Scatter(x=[-lim, lim], y=[-lim / 2, lim / 2], mode="markers", marker=dict(opacity=0, size=1), hoverinfo="skip", showlegend=False))
    fig.update_xaxes(title_text="x (m)")
    fig.update_yaxes(title_text="y (m)", scaleanchor="x", scaleratio=1)
    return _base(fig, 400)


def _two_lines(fig, xs, rows, field):
    for m, dash in (("nn", "dot"), ("gnn", "solid")):
        fig.add_trace(go.Scatter(x=xs, y=[r[m][field] for r in rows], mode="lines+markers", line=dict(color=C.COLOR_TRACK[0] if m == "nn" else C.COLOR_TRACK[1], width=3, dash=dash),
                                 name=C.METHOD_LABELS[m]))


def build_swap_chart(rows, key, xlabel, scale=1.0):
    fig = go.Figure()
    _two_lines(fig, [r[key] * scale for r in rows], rows, "swap")
    fig.update_xaxes(title_text=xlabel)
    fig.update_yaxes(title_text="Fahrten mit vertauschten Identitäten", range=[0, 1.0], tickformat=".0%")
    return _base(fig)


def build_wrong_chart(rows, key, xlabel, scale=1.0):
    fig = go.Figure()
    _two_lines(fig, [r[key] * scale for r in rows], rows, "wrong")
    fig.update_xaxes(title_text=xlabel)
    fig.update_yaxes(title_text="Falsche Zuordnungen (Anteil der Korrekturen)", range=[0, 0.6], tickformat=".0%")
    return _base(fig)
