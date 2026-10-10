"""PDF-Export der aktuellen Auswertung (fpdf2, Kernschriften; Umlaute sind dort erlaubt, Gedankenstrich und Eurozeichen nicht)."""

from fpdf import FPDF


def _num(x, nd=2):
    return f"{x:.{nd}f}".replace(".", ",")


def build_pdf(a):
    """Einseitiger Bericht: Einstellungen, Kennzahlen, Verdikt. Gibt die Bytes der PDF-Datei zurück."""
    s = a.settings
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Mehrzielverfolgung: Datenzuordnung", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, "Sebastian Hanisch, Operations Research und Machine Learning", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Einstellungen", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    rows = [
        ("Kreuzungswinkel", f"{_num(s.angle, 0)} Grad"),
        ("Zeitversatz am Kreuzungspunkt", f"{_num(s.dt, 1)} s"),
        ("Messrauschen", f"{_num(s.sigma, 1)} m"),
        ("Entdeckungswahrscheinlichkeit", f"{_num(100 * s.pd, 0)} %"),
        ("Fehlalarme je Scan", _num(s.clutter, 1)),
        ("Fahrunruhe", f"{_num(s.q, 2)} m/s²"),
        ("Zufalls-Seed", str(s.seed)),
    ]
    for k, v in rows:
        pdf.cell(95, 6, k)
        pdf.cell(0, 6, v, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Kennzahlen", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    metrics = [
        ("Kleinster Abstand der wahren Bahnen", f"{_num(a.scene.min_separation, 1)} m"),
        ("Identitäten vertauscht, nächste Messung", "ja" if a.runs["nn"].swapped else "nein"),
        ("Identitäten vertauscht, globale Zuordnung", "ja" if a.runs["gnn"].swapped else "nein"),
        ("Falsche Korrekturen, nächste Messung", f"{a.runs['nn'].wrong} von {a.runs['nn'].updates}"),
        ("Falsche Korrekturen, globale Zuordnung", f"{a.runs['gnn'].wrong} von {a.runs['gnn'].updates}"),
    ]
    for k, v in metrics:
        pdf.cell(95, 6, k)
        pdf.cell(0, 6, v, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 5, "Synthetische Fahrt mit bekanntem Modell; die Zahlen gelten für dieses Szenario und diese Einstellungen.")
    return bytes(pdf.output())
