"""Mehrzielverfolgung: welche Messung gehört zu welchem Fahrzeug? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Zwei Fahrzeuge kreuzen sich, ein Sensor liefert anonyme Positionen mit Aussetzern und Fehlalarmen. Je Fahrzeug läuft ein Kalman-Filter; die
Frage ist die Zuordnung: nächste Messung je Spur oder globale Zuordnung per Ungarischer Methode.

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import mtt_constants as C
import mtt_evaluation as ev
from mtt_evaluation import Settings, analyse
from mtt_pdf_export import build_pdf
from mtt_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from mtt_visualization import build_scene_map, build_swap_chart, build_wrong_chart

st.set_page_config(page_title="Mehrzielverfolgung – Sebastian Hanisch", layout="wide")


def de(x, nd=1):
    """Dezimalkomma statt Punkt."""
    return f"{x:.{nd}f}".replace(".", ",")


def pct(x, nd=0):
    return f"{100 * x:.{nd}f} %".replace(".", ",")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=False)
def _sweep_angle(settings):
    return ev.sweep_angle(settings)


@st.cache_data(show_spinner=False)
def _sweep_offset(settings):
    return ev.sweep_offset(settings)


@st.cache_data(show_spinner=False)
def _sweep_unrest(settings):
    return ev.sweep_unrest(settings)


@st.cache_data(show_spinner=False)
def _sweep_clutter(settings):
    return ev.sweep_clutter(settings)


def run_experiment(key, fn, settings, label):
    """Teure Sweeps nur auf Knopfdruck: das Ergebnis bleibt sichtbar, solange die Einstellungen gleich bleiben."""
    if st.button(label, key=f"run_{key}"):
        st.session_state[f"done_{key}"] = settings
    if st.session_state.get(f"done_{key}") != settings:
        st.info("Noch nicht gerechnet für diese Einstellungen. Der Knopf startet die Rechnung (einige Sekunden); danach bleibt das Ergebnis stehen.")
        return None
    with st.spinner("Rechne … (je nach Rechner einige Sekunden)"):
        return fn(settings)


st.title("🚚 Mehrzielverfolgung: welche Messung gehört zu welchem Fahrzeug?")
st.markdown(
    """
Bisher gab es ein Fahrzeug und jede Messung gehörte ihm. Mit **zwei Fahrzeugen** liefert der Sensor nur noch **anonyme Punkte**: ohne Hinweis, von wem, mit Aussetzern und gelegentlichen
Fehlalarmen. Je Fahrzeug läuft ein Kalman-Filter (**Spur**); vor jeder Korrektur muss entschieden werden, **welche Messung zu welcher Spur** gehört. Zwei Regeln werden verglichen:
die **nächste Messung je Spur** (jede Spur wählt für sich) und die **globale Zuordnung** (alle Spuren zusammen, jede Messung höchstens einer Spur, Gesamtabstand minimal; das ist ein
Zuordnungsproblem, gelöst mit der **Ungarischen Methode**). Die Demo ist das siebte Stück der Linie **Zustandsschätzung und Filter**. Im Unterschied zu den Fall-Demos zeigt sie ein einziges
Thema an einem wachsenden Beispiel.

Weiter unten: vier Experimente (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert die Zuordnung", expanded=True):
    st.markdown(
        """
Vor jeder Messrunde sagt jede Spur voraus, wo ihr Fahrzeug jetzt sein sollte, samt Unsicherheit. Für jede Spur und jede Messung gibt es einen **Abstand in Einheiten dieser Unsicherheit**
(normiertes Innovationsquadrat). Messungen außerhalb eines **Tors** (99 % der echten) kommen nicht infrage.

1. **Nächste Messung je Spur:** Jede Spur nimmt ihre nächstgelegene Messung. Zwei Spuren dürfen dieselbe nehmen.
2. **Globale Zuordnung:** Es wird die Verteilung gesucht, bei der die **Summe** aller Abstände am kleinsten ist und jede Messung höchstens einmal vorkommt (Ungarische Methode).

**Merke:** Die globale Zuordnung macht weniger falsche Zuordnungen. Ob das auch weniger **vertauschte Identitäten** bedeutet, hängt davon ab, wann die Fahrzeuge aufeinandertreffen (Experiment 2).
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Die Begegnung**")
    angle = st.slider("Kreuzungswinkel (Grad)", *bounds("angle_slider"), key="angle_slider", step=5.0,
                      help="Winkel zwischen den beiden Fahrtrichtungen. Klein heißt: die Bahnen laufen lange nahe beieinander.")
    dt = st.slider("Zeitversatz am Kreuzungspunkt (s)", *bounds("dt_slider"), key="dt_slider", step=1.0,
                   help="Das zweite Fahrzeug erreicht den Kreuzungspunkt so viele Sekunden nach dem ersten. Null heißt: beide zugleich am selben Ort.")
    q = st.slider("Fahrunruhe (m/s²)", *bounds("q_slider"), key="q_slider", step=0.05,
                  help="Beschleunigungsrauschen der Fahrzeuge (Wahrheit und Annahme der Filter). Null heißt: exakt gerade Bahnen.")
    st.markdown("**Der Sensor**")
    sigma = st.slider("Messrauschen (m)", *bounds("sigma_slider"), key="sigma_slider", step=1.0, help="Streuung der gemessenen Position je Achse.")
    pd = st.slider("Entdeckungswahrscheinlichkeit", *bounds("pd_slider"), key="pd_slider", step=0.05, format="%.2f",
                   help="Wahrscheinlichkeit, dass ein Fahrzeug in einer Sekunde überhaupt gemessen wird.")
    clutter = st.slider("Fehlalarme je Sekunde", *bounds("clutter_slider"), key="clutter_slider", step=0.5,
                        help="Im Mittel so viele Messungen je Sekunde, die zu keinem Fahrzeug gehören (zufällig im Umfeld verteilt).")
    seed = st.number_input("Zufalls-Seed der Fahrt", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Fahrt generieren", width="stretch", on_click=randomize_seed)

sync_query_params({"angle_slider": float(angle), "dt_slider": float(dt), "sigma_slider": float(sigma), "pd_slider": float(pd),
                   "clutter_slider": float(clutter), "q_slider": float(q), "seed_input": int(seed)})

settings = Settings(angle=float(angle), dt=float(dt), sigma=float(sigma), pd=float(pd), clutter=float(clutter), q=float(q), seed=int(seed))
a = _analysis(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
m1, m2, m3, m4 = st.columns(4)
for col, k in zip((m1, m3), C.METHODS):
    r = a.runs[k]
    col.metric(C.METHOD_LABELS[k], "Identitäten vertauscht" if r.swapped else "Identitäten erhalten", delta=f"{r.wrong} von {r.updates} Korrekturen falsch zugeordnet", delta_color="off")
m2.metric("Kleinster Abstand der wahren Bahnen", f"{de(a.scene.min_separation)} m", delta=f"{a.n_clutter} Fehlalarme, {a.n_missed} verpasste Entdeckungen", delta_color="off")
m4.metric("Positionsfehler der Spuren (RMSE)", f"{de(a.rmse['nn'])} m / {de(a.rmse['gnn'])} m", delta="nächste / globale Zuordnung", delta_color="off",
          help="Gegen das jeweils eigene Fahrzeug; bei vertauschten Identitäten wird er groß.")

st.markdown("**Die Begegnung** – gestrichelt: wahre Bahnen, graue Punkte: Messungen, rote Kreuze: Fehlalarme, durchgezogen: die beiden Spuren (Farbe = Identität am Start), große Punkte: Ende der Spuren.")
c1, c2 = st.columns(2)
with c1:
    st.markdown(f"**{C.METHOD_LABELS['nn']}**")
    st.plotly_chart(build_scene_map(a, "nn"), width="stretch", key="map_nn")
with c2:
    st.markdown(f"**{C.METHOD_LABELS['gnn']}**")
    st.plotly_chart(build_scene_map(a, "gnn"), width="stretch", key="map_gnn")

st.download_button("📄 Auswertung als PDF", data=build_pdf(a), file_name="mehrzielverfolgung_auswertung.pdf", mime="application/pdf", key="pdf_button")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Hilft die globale Zuordnung?")
nn, gnn = a.runs["nn"], a.runs["gnn"]
share_nn = nn.wrong / nn.updates if nn.updates else 0.0
share_gnn = gnn.wrong / gnn.updates if gnn.updates else 0.0
if nn.swapped != gnn.swapped:
    who = "die globale Zuordnung" if gnn.swapped else "die einfache Regel"
    st.warning(
        f"**Auf dieser Fahrt hat {who} die Identitäten vertauscht**, die andere nicht. Falsche Zuordnungen: nächste Messung {pct(share_nn)}, globale {pct(share_gnn)}. "
        "Eine einzelne Begegnung sagt nicht, welche Regel im Mittel besser ist: Experiment 1 bis 3 mitteln über 100 feste Begegnungen."
    )
elif nn.swapped and gnn.swapped:
    st.warning(f"**Beide Regeln haben die Identitäten vertauscht.** Falsche Zuordnungen: nächste Messung {pct(share_nn)}, globale {pct(share_gnn)}.")
else:
    st.success(f"**Beide Regeln haben die Identitäten erhalten.** Falsche Zuordnungen: nächste Messung {pct(share_nn)}, globale {pct(share_gnn)} der Korrekturen.")
st.caption("Eine einzelne Fahrt zeigt nur, was hier passiert ist; die Experimente unten mitteln über 100 feste Begegnungen.")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Wie hängt alles vom Kreuzungswinkel ab?")
st.caption(f"Anteil über {len(C.SWEEP_SEEDS)} feste Begegnungen; die übrigen Regler wie in der Seitenleiste, der Winkel läuft von 10 bis 90 Grad.")
rows = run_experiment("angle", _sweep_angle, settings, f"▶ Experiment 1 rechnen ({len(C.ANGLE_SWEEP) * len(C.SWEEP_SEEDS)} Begegnungen)")
if rows is not None:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(build_swap_chart(rows, "angle", "Kreuzungswinkel (Grad)"), width="stretch", key="angle_swap")
    with c2:
        st.plotly_chart(build_wrong_chart(rows, "angle", "Kreuzungswinkel (Grad)"), width="stretch", key="angle_wrong")
    r10, r45 = rows[0], next(r for r in rows if r["angle"] == 45.0)
    st.info(
        f"Bei {r10['angle']:.0f} Grad sind {pct(r10['nn']['wrong'])} (nächste Messung) und {pct(r10['gnn']['wrong'])} (global) der Korrekturen falsch zugeordnet, bei 45 Grad "
        f"{pct(r45['nn']['wrong'])} und {pct(r45['gnn']['wrong'])}. **Die globale Zuordnung macht bei jedem Winkel weniger falsche Zuordnungen.** Vertauschte Identitäten: "
        f"bei {r10['angle']:.0f} Grad {pct(r10['nn']['swap'])} gegen {pct(r10['gnn']['swap'])}, ab etwa 30 Grad kaum noch."
    )

st.subheader("🔬 Experiment 2: Was ändert der Zeitversatz am Kreuzungspunkt?")
st.caption(f"Anteil über {len(C.SWEEP_SEEDS)} feste Begegnungen; Winkel und übrige Regler wie in der Seitenleiste, der Zeitversatz läuft von 0 bis 6 s.")
orows = run_experiment("offset", _sweep_offset, settings, f"▶ Experiment 2 rechnen ({len(C.OFFSET_SWEEP) * len(C.SWEEP_SEEDS)} Begegnungen)")
if orows is not None:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(build_swap_chart(orows, "dt", "Zeitversatz (s)"), width="stretch", key="offset_swap")
    with c2:
        st.plotly_chart(build_wrong_chart(orows, "dt", "Zeitversatz (s)"), width="stretch", key="offset_wrong")
    o0 = orows[0]
    o4 = next(r for r in orows if r["dt"] == 4.0)
    st.info(
        f"Treffen beide **zugleich** am selben Ort (Zeitversatz 0), vertauscht die globale Zuordnung öfter ({pct(o0['gnn']['swap'])} gegen {pct(o0['nn']['swap'])}); mit **4 s** Zeitversatz "
        f"kehrt es sich um ({pct(o4['gnn']['swap'])} gegen {pct(o4['nn']['swap'])}). Erklärung: Wenn beide Fahrzeuge denselben Ort treffen, greift die einfache Regel für beide Spuren dieselbe Messung "
        "und lässt beide mit ihrer Geschwindigkeit weiterlaufen; die globale Zuordnung muss die beiden Messungen auf die Spuren verteilen und kann dabei zufällig falsch liegen."
    )

st.subheader("🔬 Experiment 3: Und wenn die Fahrzeuge nicht geradeaus fahren?")
st.caption(f"Anteil über {len(C.SWEEP_SEEDS)} feste Begegnungen; Winkel, Zeitversatz und übrige Regler wie in der Seitenleiste, die Fahrunruhe läuft von 0 bis 0,5 m/s².")
urows = run_experiment("unrest", _sweep_unrest, settings, f"▶ Experiment 3 rechnen ({len(C.Q_SWEEP) * len(C.SWEEP_SEEDS)} Begegnungen)")
if urows is not None:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(build_swap_chart(urows, "q", "Fahrunruhe (m/s²)"), width="stretch", key="unrest_swap")
    with c2:
        st.plotly_chart(build_wrong_chart(urows, "q", "Fahrunruhe (m/s²)"), width="stretch", key="unrest_wrong")
    q0 = urows[0]
    q3 = next(r for r in urows if r["q"] == 0.3)
    st.info(
        f"Bei exakt geraden Bahnen (0 m/s²) behält die einfache Regel die Identitäten in {pct(1 - q0['nn']['swap'])} der Begegnungen, die globale in {pct(1 - q0['gnn']['swap'])}; "
        f"bei 0,3 m/s² sind es {pct(1 - q3['nn']['swap'])} und {pct(1 - q3['gnn']['swap'])}: sobald die Geschwindigkeit als Gedächtnis weniger trägt, "
        "hilft die globale Zuordnung."
    )

st.subheader("🔬 Experiment 4: Und mit Fehlalarmen?")
st.caption(f"Anteil über {len(C.SWEEP_SEEDS)} feste Begegnungen; übrige Regler wie in der Seitenleiste, die Fehlalarme laufen von 0 bis 3 je Sekunde.")
crows = run_experiment("clutter", _sweep_clutter, settings, f"▶ Experiment 4 rechnen ({len(C.CLUTTER_SWEEP) * len(C.SWEEP_SEEDS)} Begegnungen)")
if crows is not None:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(build_swap_chart(crows, "clutter", "Fehlalarme je Sekunde"), width="stretch", key="clutter_swap")
    with c2:
        st.plotly_chart(build_wrong_chart(crows, "clutter", "Fehlalarme je Sekunde"), width="stretch", key="clutter_wrong")
    c0, c3 = crows[0], crows[-1]
    st.info(
        f"Von 0 auf {c3['clutter']:.0f} Fehlalarme je Sekunde ändern sich die falschen Zuordnungen kaum (nächste Messung {pct(c0['nn']['wrong'])} → {pct(c3['nn']['wrong'])}, global "
        f"{pct(c0['gnn']['wrong'])} → {pct(c3['gnn']['wrong'])}), weil das Tor die weit entfernten Fehlalarme ausschließt: **in diesem Aufbau (150 m Umfeld, 99-%-Tor) sind Fehlalarme nicht der Engpass**."
    )

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Jede Entscheidung gilt endgültig** | Beide Regeln legen sich je Sekunde fest. Treffen die Fahrzeuge zugleich am selben Ort, ist die richtige Entscheidung erst später erkennbar. | Mehrere Hypothesen über mehrere Sekunden (MHT), gewichtete Zuordnung (JPDA) |
| **Die Zahl der Fahrzeuge ist bekannt** | Spuren werden hier mit bekannter Identität gestartet; wie sie entstehen und enden, ist nicht Teil der Demo. | Spurverwaltung |
| **Fehlalarme sind gleichverteilt und selten** | Strukturierte Fehlalarme (Reflexe nahe der Fahrzeuge) liegen im Tor und werden zugeordnet. | Modelle für Fehlalarme |
| **Bewegungsmodell stimmt** | Bei Manövern liegen die Vorhersagen daneben, und beide Zuordnungsregeln verfehlen. | Mehrere Bewegungsmodelle (IMM) |
| **Messungen sind unabhängig und gaußsch** | Systematische Fehler verschieben beide Messungen gemeinsam. | Robuste Filter |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Spuren.** Je Fahrzeug $i$ ein Kalman-Filter für $x^i = (x, y, v_x, v_y)^\top$ wie in den vorigen Stücken; Messung der Position mit Varianz $r = \sigma^2$ je Achse.

**Abstand (Kosten).** Für Spur $i$ und Messung $j$ nach der Vorhersage: $\nu_{ij} = z_j - H\hat x^i$, $S^i = H P^i H^\top + rI$ und
$$c_{ij} = \nu_{ij}^\top (S^i)^{-1} \nu_{ij}.$$
Tor: nur $c_{ij} \le \gamma$ mit $\gamma = \chi^2_{2;\,0{,}99} \approx 9{,}21$.

**Nächste Messung.** $j(i) = \arg\min_j c_{ij}$ unter dem Tor, für jede Spur unabhängig (dieselbe Messung darf mehrfach vorkommen).

**Globale Zuordnung.** Lineares Zuordnungsproblem: minimiere $\sum_{i} c_{i,\pi(i)}$ über injektive Abbildungen $\pi$ von Spuren auf Messungen (oder auf „keine Messung“ zu den Kosten $\gamma$).
Gelöst mit der **Ungarischen Methode** (Kuhn–Munkres, mit Potentialen), Aufwand $O(n^2 m)$.

**Vertauschte Identitäten.** Am Ende liegen die Spurpositionen näher an dem jeweils anderen Fahrzeug: $\lVert p_1 - t_2\rVert + \lVert p_2 - t_1\rVert < \lVert p_1 - t_1\rVert + \lVert p_2 - t_2\rVert$.

**Literatur.** H. W. Kuhn (1955), *The Hungarian method for the assignment problem*, Naval Research Logistics Quarterly 2(1–2), 83–97; J. Munkres (1957), *Algorithms for the assignment and transportation
problems*, Journal of the Society for Industrial and Applied Mathematics 5(1), 32–38.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
