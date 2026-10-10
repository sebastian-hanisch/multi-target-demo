"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

# Zeitachse: ein Schritt = eine Sekunde, 60 Schritte; die Fahrzeuge treffen sich bei Schritt T/2
DT = 1.0
T_STEPS = 60
SPEED = 10.0                      # Geschwindigkeit beider Fahrzeuge in m/s
Q_FLOOR = 0.05                    # kleinste Beschleunigungsrauschen-Annahme des Filters (numerisch)
V_INIT_STD = 1.0                  # Fehler der bekannten Anfangsgeschwindigkeit je Achse (m/s)
GATE_PROB_CHI2 = 9.2103           # Tor: 99-%-Quantil der Chi-Quadrat-Verteilung mit 2 Freiheitsgraden
CLUTTER_BOX = 150.0               # Fehlalarme sind gleichverteilt im Quadrat +-CLUTTER_BOX um den Kreuzungspunkt
TRACKS = 2
SWAP_OK_M = 0.0

# Reglergrenzen
ANGLE_MIN, ANGLE_MAX, DEFAULT_ANGLE = 5.0, 90.0, 10.0           # Kreuzungswinkel in Grad
DT_MIN, DT_MAX, DEFAULT_DT = 0.0, 6.0, 0.0                      # Zeitversatz am Kreuzungspunkt in s
SIGMA_MIN, SIGMA_MAX, DEFAULT_SIGMA = 3.0, 25.0, 15.0           # Messrauschen in m
PD_MIN, PD_MAX, DEFAULT_PD = 0.5, 1.0, 0.9                      # Entdeckungswahrscheinlichkeit
CLUTTER_MIN, CLUTTER_MAX, DEFAULT_CLUTTER = 0.0, 3.0, 0.0       # mittlere Fehlalarme je Scan
Q_MIN, Q_MAX, DEFAULT_Q = 0.0, 0.5, 0.1                         # Fahrunruhe der Fahrzeuge in m/s^2
SEED_MAX, DEFAULT_SEED = 999999, 35

# feste Instanzen für alle Sweeps: nie ein einzelner Seed als Befund
SWEEP_SEEDS = tuple(range(100000, 100100))
ANGLE_SWEEP = (10.0, 20.0, 30.0, 45.0, 60.0, 90.0)
CLUTTER_SWEEP = (0.0, 0.5, 1.0, 2.0, 3.0)
PD_SWEEP = (1.0, 0.9, 0.7, 0.5)
OFFSET_SWEEP = (0.0, 1.0, 2.0, 4.0, 6.0)
Q_SWEEP = (0.0, 0.1, 0.3, 0.5)

METHODS = ("nn", "gnn")
METHOD_LABELS = {"nn": "Nächste Messung je Spur", "gnn": "Globale Zuordnung (Ungarisch)"}

COLOR_TRUTH = ("#2F6B65", "#8a5a9a")
COLOR_TRACK = ("#4c78a8", "#c97b2a")
COLOR_RAW = "rgba(150,150,150,0.8)"
COLOR_CLUTTER = "#b04a4a"

PRESETS = {
    "Standard (Voreinstellung)": {"angle": DEFAULT_ANGLE, "dt": DEFAULT_DT, "sigma": DEFAULT_SIGMA, "pd": DEFAULT_PD, "clutter": DEFAULT_CLUTTER, "q": DEFAULT_Q, "seed": DEFAULT_SEED},
    "Zeitversatz am Kreuzungspunkt": {"angle": DEFAULT_ANGLE, "dt": 4.0, "sigma": DEFAULT_SIGMA, "pd": DEFAULT_PD, "clutter": DEFAULT_CLUTTER, "q": DEFAULT_Q, "seed": DEFAULT_SEED},
    "Mit Fehlalarmen": {"angle": 20.0, "dt": 0.0, "sigma": DEFAULT_SIGMA, "pd": DEFAULT_PD, "clutter": 2.0, "q": DEFAULT_Q, "seed": DEFAULT_SEED},
    "Steiler Winkel": {"angle": 60.0, "dt": 0.0, "sigma": DEFAULT_SIGMA, "pd": DEFAULT_PD, "clutter": DEFAULT_CLUTTER, "q": DEFAULT_Q, "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Standard (Voreinstellung)": "Zwei Fahrzeuge kreuzen sich unter 10 Grad, beide zugleich am Kreuzungspunkt, GPS-Rauschen 15 m, jede zehnte Messung fehlt, keine Fehlalarme: die schwerste Lage für die Zuordnung.",
    "Zeitversatz am Kreuzungspunkt": "Das zweite Fahrzeug erreicht den Kreuzungspunkt 4 Sekunden nach dem ersten: sie sind nie am selben Ort zugleich.",
    "Mit Fehlalarmen": "Zwei Fehlalarme je Scan bei 20 Grad: zusätzliche Messungen, die zu keinem Fahrzeug gehören.",
    "Steiler Winkel": "60 Grad: die Fahrzeuge sind schnell wieder auseinander, Verwechslungen sind selten.",
}
