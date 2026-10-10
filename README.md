# Mehrzielverfolgung: welche Messung gehört zu welchem Fahrzeug? – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-multi-target-demo.streamlit.app/)** -->

Siebtes Stück der Linie **Zustandsschätzung und Filter** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“.
Es setzt [Kalman-Filter](https://github.com/sebastian-hanisch/kalman-filter-demo) fort und behebt dessen benennbare Schwäche **ein Fahrzeug, klare Zuordnung**: bei zwei Fahrzeugen
liefert der Sensor nur anonyme Punkte, und vor jeder Korrektur muss entschieden werden, welche Messung zu welcher Spur gehört. Verglichen werden **die nächste Messung je Spur** und die
**globale Zuordnung** per Ungarischer Methode (ein lineares Zuordnungsproblem, wie in der Matching-Linie). Im Unterschied zu den Fall-Demos zeigt die Demo ein einziges Thema an einem wachsenden Beispiel.

## Warum dieses Problem

Sobald mehrere Objekte verfolgt werden, ist die Zuordnung der Messungen der eigentliche Engpass, nicht die Filterung. Naheliegend wäre, dass die „global bessere“ Zuordnung auch die Identitäten
sicherer erhält. Die Messung zeigt etwas Differenzierteres.

## Modell

Zwei Fahrzeuge mit 10 m/s kreuzen sich unter einem wählbaren Winkel in der Mitte der Fahrt (60 s); das zweite erreicht den Kreuzungspunkt optional einige Sekunden später. Der Sensor liefert je Sekunde
und Fahrzeug mit der Entdeckungswahrscheinlichkeit eine Position mit Gauß-Rauschen, dazu Fehlalarme (Poisson, gleichverteilt im 300-m-Quadrat) in zufälliger Reihenfolge, ohne Identität. Je Fahrzeug ein
Kalman-Filter mit konstanter Geschwindigkeit; Spuren starten mit bekannter Identität. Abstand = normiertes Innovationsquadrat, Tor = 99-%-Quantil von Chi-Quadrat mit 2 Freiheitsgraden. Die
Ungarische Methode ist selbst geschrieben (numpy). Formeln im 📐-Expander, Kern in `mtt_algorithm.py`. Die Bahnen sind gerade Linien mit kleiner Beschleunigungsstörung; die Filter kennen deren Größe.

## Methodik

Vier Experimente auf Knopfdruck, je über 100 feste Begegnungen (`SWEEP_SEEDS`):

1. Kreuzungswinkel von 10 bis 90 Grad: vertauschte Identitäten und falsche Zuordnungen.
2. Zeitversatz am Kreuzungspunkt von 0 bis 6 s.
3. Fahrunruhe von 0 bis 0,5 m/s².
4. Fehlalarme von 0 bis 3 je Sekunde.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: Anteil über 100 feste Begegnungen, Standardeinstellung (10 Grad, GPS 15 m, Entdeckung 90 %, Fahrunruhe 0,1 m/s²); `tests/test_claims.py` rechnet jede nach.

| Frage | Befund |
|---|---|
| Wer macht weniger falsche Zuordnungen? | Die **globale Zuordnung bei jedem Winkel**: bei 10 Grad **26 %** der Korrekturen falsch gegen **38 %**, bei 45 Grad **4 %** gegen **8 %**. |
| Wer vertauscht seltener die Identitäten? | Hängt vom **Zeitversatz** ab. Beide zugleich am Kreuzungspunkt (10 Grad): globale Zuordnung **30 %**, nächste Messung **13 %**. Mit 4 s Zeitversatz: globale Zuordnung **6 %**, nächste Messung **28 %**. |
| Und bei Fahrunruhe? | Bei exakt geraden Bahnen (0 m/s²) behält die einfache Regel die Identitäten nahezu immer (Vertauschungen unter 3 %), die globale in 77 %; bei 0,3 m/s² vertauscht die einfache Regel **49 %**, die globale **30 %**. |
| Ab welchem Winkel ist es kein Problem mehr? | Ab etwa 45 Grad vertauscht keine der Regeln mehr als 3 % der Begegnungen. |
| Fehlalarme | Von 0 auf 3 je Sekunde ändern sich falsche Zuordnungen und Vertauschungen nicht nennenswert (unter 6 bzw. 10 Prozentpunkte): das Tor schließt die weit entfernten Fehlalarme aus. In diesem Aufbau sind sie nicht der Engpass. |

## Ehrliche Grenzen

- **Die Wahrheit ist das Modell.** Zwei Fahrzeuge auf fast geraden Bahnen, Gauß-Rauschen, bekannte Zahl; kein echter Sensordatensatz.
- **Die Vertauschung wird nur am Ende gezählt.** Eine zwischenzeitliche Vertauschung, die sich wieder aufhebt, zählt nicht; die Kennzahl misst, ob die Spuren am Ende bei ihren Fahrzeugen sind.
- **Zugleich am selben Ort ist ein Sonderfall.** Bei Zeitversatz 0 sind beide Fahrzeuge zu einem Zeitpunkt exakt am selben Ort; dann ist die Zuordnung dieser Sekunde nicht entscheidbar.
- **Der Befund ist nicht „die einfache Regel ist besser“.** Er ist: die globale Zuordnung korrigiert falsche Zuordnungen, aber die Identitäten hängen an der Geschwindigkeitsschätzung, und die globale Zuordnung kann diese bei gleichzeitigem Aufeinandertreffen und exakt geraden Bahnen schlechter erhalten.
- **Fehlalarme sind hier harmlos wegen des Aufbaus** (gleichverteilt, 300-m-Quadrat, 99-%-Tor); strukturierte oder dichte Fehlalarme sind nicht gezeigt.
- **JPDA, MHT:** mehrere Hypothesen über mehrere Sekunden sind nicht umgesetzt.
- **Kleine Anteile sind Zufall:** bei 100 Begegnungen betragen einzelne Prozentpunkte beim Anteil vertauschter Identitäten nicht viel.

## Verifikation

- **Kern:** Zuordnung (Handbeispiele, Tor, keine Messungen, mehr Spuren als Messungen), Abstandsmatrix, Korrektur, Vertauschungsdefinition, Poisson-Zufallszahl (monoton), Verschachtelung der Regler (`tests/test_algorithm.py`).
- **Orakel** (`tests/test_oracle_multi_target.py`, anderer Rechenweg):
  - **Ungarische Methode** gegen `scipy.optimize.linear_sum_assignment` und gegen Aufzählung aller Zuordnungen (150 Zufallsmatrizen, auch mit Gleichständen).
  - **Globale Zuordnung mit Tor und „keine Messung“** gegen Aufzählung aller zulässigen Zuordnungen.
  - Tracker gleich dem Kalman-Filter mit **bekannter Identität**, wo alle Korrekturen stattfinden.
  - Tor-Rate auf echten Messungen gleich 1 − Quantil; Poisson-Mittel der Fehlalarme.
  - Angezeigte Kennzahlen gegen unabhängige Neuberechnung.
- **Zahlen im README:** `tests/test_claims.py`.
- **Oberfläche:** AppTest-Rauchtest (jeder Button, Regler an Minimum und Maximum, Experiment-Knöpfe) und Footer-Test.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App (Skelett nach DEMO-PLAYBOOK §2) |
| `mtt_algorithm.py` | Kern: Kalman-Korrektur, Abstandsmatrix, Ungarische Methode, Zuordnungsregeln |
| `mtt_scenario.py` | Bahnen, anonyme Messungen mit Aussetzern und Fehlalarmen (ein Generator, Regler schwellen denselben Zufall) |
| `mtt_evaluation.py` | Tracker, Kennzahlen, Sweeps |
| `mtt_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets |
| `mtt_presets.py` | Permalink (Begrenzen), Presets, Zufalls-Seed-Button |
| `mtt_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `mtt_pdf_export.py` | PDF-Auswertung (fpdf2) |
| `requirements.txt`, `requirements-dev.txt` | Laufzeit bzw. Laufzeit plus pytest/scipy |
| `tests/` | Einheiten, Orakel, Behauptungen, Presets, AppTest, Footer |

## Bewusst nicht umgesetzt

- **Mehr als zwei Fahrzeuge**, Spurstart und -ende, wechselnde Zahl der Ziele.
- **JPDA, MHT** und andere Verfahren mit mehreren Hypothesen.
- **Echte Sensordaten.**

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
