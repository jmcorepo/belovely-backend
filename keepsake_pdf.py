"""Belovely Keepsake — renders a song's lyrics into a printable, framed PDF.

Pure-reportlab (no system libs) so it builds cleanly in the Render Docker image.
Called from the orders/paid webhook when the customer bought the $19 Keepsake bump.
Avoids glyphs outside WinAnsi (e.g. ♫/♥) so the standard Type-1 fonts render them.
"""
from __future__ import annotations

import html as _html
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

CREAM = HexColor("#FBF7F0")
INK = HexColor("#2A241F")
GOLD = HexColor("#C9A24B")
MUTED = HexColor("#8A7B66")


def _esc(s) -> str:
    return _html.escape(str(s), quote=False)


def _sections(plan: dict):
    """[(label, [lines]), ...] — skips instrumental sections that have no lyrics."""
    out = []
    for s in (plan.get("sections") or []):
        lines = [str(ln).strip() for ln in (s.get("lines") or []) if str(ln).strip()]
        if not lines:
            continue
        label = str(s.get("section_name") or "").strip()
        out.append((label, lines))
    return out


def _page(canvas, doc):
    w, h = letter
    canvas.saveState()
    canvas.setFillColor(CREAM)
    canvas.rect(0, 0, w, h, fill=1, stroke=0)
    # double gold frame
    canvas.setStrokeColor(GOLD)
    canvas.setLineWidth(1.4)
    m = 0.5 * inch
    canvas.rect(m, m, w - 2 * m, h - 2 * m, fill=0, stroke=1)
    canvas.setLineWidth(0.6)
    canvas.rect(m + 6, m + 6, w - 2 * m - 12, h - 2 * m - 12, fill=0, stroke=1)
    # footer wordmark
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 9)
    canvas.drawCentredString(w / 2, m + 16, "BELOVELY   ·   belovelygifts.com")
    canvas.restoreState()


def build_lyrics_pdf(out_path, recipient_name: str, genre: str, plan: dict,
                     message: str | None = None) -> str:
    """Render the lyrics keepsake to out_path (PDF). Returns the path as str."""
    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    title = ParagraphStyle("t", fontName="Times-Bold", fontSize=30, leading=34,
                           textColor=INK, alignment=TA_CENTER, spaceAfter=2)
    subtitle = ParagraphStyle("s", fontName="Helvetica", fontSize=10.5, leading=15,
                              textColor=MUTED, alignment=TA_CENTER, spaceAfter=22)
    label = ParagraphStyle("l", fontName="Helvetica-Bold", fontSize=10, leading=13,
                           textColor=GOLD, alignment=TA_CENTER, spaceBefore=16, spaceAfter=7)
    line = ParagraphStyle("ln", fontName="Times-Roman", fontSize=13.5, leading=21,
                          textColor=INK, alignment=TA_CENTER)
    dedication = ParagraphStyle("d", fontName="Times-Italic", fontSize=12, leading=18,
                                textColor=MUTED, alignment=TA_CENTER, spaceBefore=18)

    name = (recipient_name or "Your loved one").strip()
    g = (genre or "song").strip()

    story = [
        Spacer(1, 10),
        Paragraph(f"{_esc(name)}’s Song", title),
        Paragraph(_esc(f"A one-of-a-kind {g} · written with love by Belovely").upper(), subtitle),
    ]
    for lbl, lines in _sections(plan):
        if lbl:
            story.append(Paragraph(_esc(lbl).upper(), label))
        for ln in lines:
            story.append(Paragraph(_esc(ln), line))
        story.append(Spacer(1, 6))
    if message and str(message).strip():
        story.append(Paragraph(f"“{_esc(str(message).strip())}”", dedication))

    doc = SimpleDocTemplate(
        out_path, pagesize=letter,
        leftMargin=1.15 * inch, rightMargin=1.15 * inch,
        topMargin=1.25 * inch, bottomMargin=1.1 * inch,
        title=f"{name}'s Song — Belovely Keepsake", author="Belovely",
    )
    doc.build(story, onFirstPage=_page, onLaterPages=_page)
    return out_path
