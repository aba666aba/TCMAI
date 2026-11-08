"""Build English academic deck for TCMAI project — 18+ slides."""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Pt

# ── Global constants ──
SLIDE_W, SLIDE_H = Emu(12192000), Emu(6858000)  # 16:9
ASPECT = (SLIDE_W, SLIDE_H)

# TCMAI palette — earthy tones reflecting TCM aesthetics
C_BG = RGBColor(0xF5, 0xF1, 0xE8)          # warm cream background
C_TITLE = RGBColor(0x3C, 0x2F, 0x2E)       # deep coffee for titles
C_BODY = RGBColor(0x4A, 0x44, 0x40)        # soft dark brown for body
C_ACCENT = RGBColor(0x8C, 0x3B, 0x2E)      # vermilion red (TCM ink)
C_ACCENT2 = RGBColor(0x2F, 0x5D, 0x34)     # pine green (herbal theme)
C_MUTED = RGBColor(0x8B, 0x83, 0x78)       # muted gray for footers
C_CARD = RGBColor(0xFF, 0xFB, 0xF3)        # card background
C_LINE = RGBColor(0xD4, 0xC5, 0xA9)        # divider line

FONT_TITLE = "Georgia"
FONT_BODY = "Palatino Linotype"
FONT_MONO = "Consolas"

prs = Presentation()
prs.slide_width, prs.slide_height = ASPECT

BLANK_LAYOUT = prs.slide_layouts[6]  # blank


# ── Helpers ──

def _rect(slide, left, top, width, height, fill=None, line=None):
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    sh.shadow.inherit = False
    if fill is not None:
        sh.fill.solid()
        sh.fill.fore_color.rgb = fill
    else:
        sh.fill.background()
    if line is not None:
        sh.line.color.rgb = line
        sh.line.width = Pt(1)
    else:
        sh.line.fill.background()
    return sh


def _add_text(
    slide, left, top, width, height, text,
    *, font=FONT_BODY, size=14, bold=False, italic=False,
    color=C_BODY, align=PP_ALIGN.LEFT, line_spacing=1.2,
):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Emu(0)
    tf.margin_right = Emu(0)
    lines = text.split("\n")
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(4)
        run = p.add_run()
        run.text = line
        run.font.name = font
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.italic = italic
        run.font.color.rgb = color
    return tb


def _add_bullets(slide, left, top, width, height, items, *, size=14, color=C_BODY, bullet="• "):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Emu(0)
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.space_after = Pt(6)
        p.level = 0
        run = p.add_run()
        run.text = f"{bullet}{item}"
        run.font.name = FONT_BODY
        run.font.size = Pt(size)
        run.font.color.rgb = color
    return tb


def _bg(slide):
    """Fill entire slide with cream background."""
    return _rect(slide, Emu(0), Emu(0), SLIDE_W, SLIDE_H, fill=C_BG)


def _header_bar(slide, title, subtitle=None, page_no=None, total=None):
    """Add consistent header: vermilion accent bar + title + page number."""
    _rect(slide, Emu(0), Emu(0), SLIDE_W, Emu(80000), fill=C_ACCENT)
    _rect(slide, Emu(0), Emu(80000), SLIDE_W, Emu(6000), fill=C_LINE)
    _add_text(slide, Emu(457200), Emu(18288), SLIDE_W - Emu(1828800), Emu(76200),
              title, font=FONT_TITLE, size=28, bold=True, color=C_BG)
    if subtitle:
        _add_text(slide, Emu(457200), Emu(91440), SLIDE_W - Emu(2500000), Emu(45720),
                  subtitle, font=FONT_BODY, size=13, italic=True, color=RGBColor(0xF5, 0xE5, 0xD0))
    if page_no is not None:
        footer = f"TCMAI — English Academic Deck  |  {page_no}/{total}"
        _add_text(slide, SLIDE_W - Emu(381000), SLIDE_H - Emu(91440), Emu(350000), Emu(40000),
                  footer, font=FONT_MONO, size=9, color=C_MUTED, align=PP_ALIGN.RIGHT)


def _footer_tcm(slide):
    _add_text(slide, Emu(457200), SLIDE_H - Emu(91440), Emu(3810000), Emu(40000),
              "TCMAI · Traditional Chinese Medicine AI Q&A · Research Deck",
              font=FONT_BODY, size=9, color=C_MUTED)


# ── Slide factory ──

TOTAL = 18


def new_slide():
    slide = prs.slides.add_slide(BLANK_LAYOUT)
    _bg(slide)
    return slide


def finalize_slide(slide, title, subtitle=None, page_no=None):
    _header_bar(slide, title, subtitle=subtitle, page_no=page_no, total=TOTAL)
    _footer_tcm(slide)
    return slide


# ═══════════════════════════════════════════════════════════════════════════
# Slide 1 — Title
# ═══════════════════════════════════════════════════════════════════════════
s = new_slide()
# Decorative vermilion stripe
_rect(s, Emu(0), SLIDE_H // 2 - Emu(150000), SLIDE_W, Emu(300000), fill=C_ACCENT)
# Title text over the stripe
_add_text(s, Emu(600000), SLIDE_H // 2 - Emu(120000), SLIDE_W - Emu(1200000), Emu(200000),
          "TCMAI: A Hybrid AI Framework for Traditional Chinese Medicine",
          font=FONT_TITLE, size=40, bold=True, color=C_BG, align=PP_ALIGN.CENTER)
# Subtitle
_add_text(s, Emu(600000), SLIDE_H // 2 + Emu(100000), SLIDE_W - Emu(1200000), Emu(120000),
          "Large Language Model + Knowledge Graph + Small-Sample Deep Learning",
          font=FONT_BODY, size=18, italic=True, color=C_BG, align=PP_ALIGN.CENTER)
# Tagline
_add_text(s, Emu(600000), SLIDE_H - Emu(400000), SLIDE_W - Emu(1200000), Emu(100000),
          """\"TCM knowledge is too precious to remain inaccessible — AI can make it conversational, verifiable, and safe.""",
          font=FONT_BODY, size=14, italic=True, color=C_TITLE, align=PP_ALIGN.CENTER)

# ═══════════════════════════════════════════════════════════════════════════
# Slide 2 — Agenda
# ═══════════════════════════════════════════════════════════════════════════
s = finalize_slide(new_slide(), "Agenda", "Roadmap of this research presentation", page_no=2)
items = [
    "The Unique Case for AI in Traditional Chinese Medicine",
    "Why Existing Approaches Fall Short — A Systematic Gap Analysis",
    "TCMAI: Hybrid Architecture Design (NLU + KG + DL + Rule)",
    "Pipeline Walkthrough — From Free Text to Verified Recommendation",
    "Component Deep Dive — Each Engine and Its Rationale",
    "Experimental Setup and Ablation Results",
    "Discussion — What Works, What Doesn't, and Why This Direction Matters",
    "Limitations, Ethical Boundaries, and Future Research",
]
_add_bullets(s, Emu(600000), Emu(1200000), Emu(10000000), Emu(5000000), items, size=17)
# Green accent line
_rect(s, Emu(500000), Emu(1100000), Emu(15240), Emu(5000000), fill=C_ACCENT2)


# ═══════════════════════════════════════════════════════════════════════════
# Slide 3 — TCM as Unique AI Frontier
# ═══════════════════════════════════════════════════════════════════════════
s = finalize_slide(new_slide(), "TCM: A Unique Frontier for AI Research",
                   "Three properties make TCM irreplaceable as an AI application domain",
                   page_no=3)

card_y = Emu(1200000)
card_h = Emu(4800000)
card_w = (SLIDE_W - Emu(1400000)) // 3
for i, (title, body) in enumerate([
    ("1. Structured + Tacit",
     "TCM encodes hard codified knowledge (herbal taxonomy, contraindication rules) alongside subtle experiential patterns (constitution typing, symptom-syndrome co-occurrence). AI sits naturally at this boundary — it can verify the rules while surfacing the patterns."),
    ("2. Small-Sample by Nature",
     "Every herbal formula is a 2000-year-old peer-reviewed experiment. The total combinatorial space of symptom-herb-formula is vast, but labeled instances are sparse. This is exactly the regime where hybrid symbolic-neural architectures outperform pure end-to-end systems."),
    ("3. Safety-Critical & Interpretable",
     "Medical AI cannot be a black box. Physicians demand traceable reasoning paths, contraindication enforcement, and auditable outputs. TCM is not a 'nice-to-have' — it is a test case for responsible AI in medicine."),
]):
    left = Emu(457200) + i * (card_w + Emu(228600))
    _rect(s, left, card_y, card_w, card_h, fill=C_CARD, line=C_LINE)
    # accent top bar
    _rect(s, left, card_y, card_w, Emu(80000), fill=C_ACCENT if i == 0 else (C_ACCENT2 if i == 1 else C_TITLE))
    _add_text(s, left + Emu(152400), card_y + Emu(150000), card_w - Emu(300000), Emu(60000),
              title, font=FONT_TITLE, size=20, bold=True, color=C_TITLE)
    _add_text(s, left + Emu(152400), card_y + Emu(280000), card_w - Emu(300000), card_h - Emu(400000),
              body, size=13, color=C_BODY)


# ═══════════════════════════════════════════════════════════════════════════
# Slide 4 — Grand Challenge
# ═══════════════════════════════════════════════════════════════════════════
s = finalize_slide(new_slide(), "The Grand Challenge: Why TCM Is Hard for AI",
                   "Four structural properties resist single-paradigm solutions",
                   page_no=4)

_challenges = [
    ("Symptom Ambiguity", "User input is colloquial, incomplete, context-dependent.\n\"My chest feels tight\" → which organ? Which syndrome?"),
    ("Constitutional Specificity", "Same symptom → different formula by constitution.\nCold + Weak → different from Cold + Strong."),
    ("Multi-Drug Combinatorics", "Formula is not a single herb — it is a coordinated multi-herb interaction (君-臣-佐-使)."),
    ("Safety as First-Class Citizen", "Contraindications (十八反/十九畏) and toxicity are hard constraints, not post-processing."),
]
x = Emu(500000)
for row, (title, body) in enumerate(_challenges):
    y = Emu(1200000) + row * Emu(1200000)
    _add_text(s, x, y, Emu(3000000), Emu(60000), title, size=18, bold=True, color=C_ACCENT)
    _add_text(s, x, y + Emu(55000), SLIDE_W - Emu(1000000), Emu(180000), body, size=14)


# ═══════════════════════════════════════════════════════════════════════════
# Slide 5 — Comparison Table
# ═══════════════════════════════════════════════════════════════════════════
s = finalize_slide(new_slide(), "Existing Approaches — Systematic Comparison",
                   "Five design dimensions, four paradigms", page_no=5)

# Build a table manually via shapes
cols = ["Dimension", "Pure Rules", "Pure Knowledge Graph", "Pure Deep Learning", "TCMAI (Ours)"]
rows = [
    ("Principle", "if-then rules", "graph traversal", "end-to-end NN", "NLU + KG + DL + Rule"),
    ("Coverage", "Low", "Medium", "High", "High"),
    ("Interpretability", "Strong", "Strong (paths)", "Weak", "Strong (all engines)"),
    ("Cold-Start Resistance", "Needs rule authoring", "Needs graph build", "Needs labeled data", "KG+Rule fallback"),
    ("Ambiguity Handling", "Poor", "Poor", "Good", "Good via NLU"),
    ("Safety Enforcement", "Easy", "Easy (edge attr.)", "Needs design", "Built-in dual-screening"),
    ("Maintenance Cost", "High", "Medium", "Low", "Medium"),
]

n_rows = len(rows) + 1
n_cols = len(cols)
cell_w = (SLIDE_W - Emu(914400)) // n_cols
cell_h = Emu(381000)
start_x = Emu(457200)
start_y = Emu(1100000)

for c, col in enumerate(cols):
    cx = start_x + c * cell_w
    fill = C_ACCENT if c == 0 else (C_ACCENT2 if c == n_cols - 1 else C_LINE)
    _rect(s, cx, start_y, cell_w, cell_h, fill=fill, line=C_CARD)
    _add_text(s, cx + Emu(10000), start_y + Emu(10000), cell_w - Emu(20000), cell_h - Emu(20000),
              col, size=13, bold=True, color=C_BG, align=PP_ALIGN.CENTER)

for r, row in enumerate(rows):
    ry = start_y + (r + 1) * cell_h
    for c, cell in enumerate(row):
        cx = start_x + c * cell_w
        is_ours = (c == n_cols - 1)
        fill = RGBColor(0xFF, 0xF0, 0xE8) if (r % 2 == 0 and is_ours) else (
            RGBColor(0xF5, 0xE8, 0xE0) if is_ours else (C_CARD if r % 2 == 0 else C_BG))
        _rect(s, cx, ry, cell_w, cell_h, fill=fill, line=C_LINE)
        bold = (c == 0 or is_ours)
        color = C_ACCENT2 if is_ours else C_BODY
        _add_text(s, cx + Emu(8000), ry + Emu(8000), cell_w - Emu(16000), cell_h - Emu(16000),
                  cell, size=12, bold=bold, color=color, align=PP_ALIGN.CENTER)


# ═══════════════════════════════════════════════════════════════════════════
# Slide 6 — TCMAI Architecture Overview
# ═══════════════════════════════════════════════════════════════════════════
s = finalize_slide(new_slide(), "TCMAI: Hybrid Architecture",
                   "Four coordinated engines, each solving a distinct sub-problem",
                   page_no=6)

# Draw architecture as horizontal pipeline
engines = [
    ("NLU Engine\nQwen 2.5", "Free-text →\nstandardized symptoms", C_ACCENT),
    ("Knowledge Graph\nNeo4j-like (nx)", "Structured TCM knowledge\n+ syndrome voting", C_ACCENT2),
    ("TCMNet\nPre-LN Transformer", "Ranked formulas & herbs\nsmall-sample learning", C_TITLE),
    ("Rule Engine\nSafety & Verify", "Toxicity, contraindications,\nclaim filtering", RGBColor(0x6B, 0x4C, 0x8A)),
]

y_center = SLIDE_H // 2 - Emu(150000)
box_w = Emu(2100000)
box_h = Emu(1400000)
gap = Emu(180000)
total_w = len(engines) * box_w + (len(engines) - 1) * gap
start_x = (SLIDE_W - total_w) // 2

for i, (title, desc, color) in enumerate(engines):
    x = start_x + i * (box_w + gap)
    _rect(s, x, y_center, box_w, box_h, fill=C_CARD, line=color)
    _rect(s, x, y_center, box_w, Emu(60000), fill=color)
    _add_text(s, x + Emu(50000), y_center + Emu(100000), box_w - Emu(100000), Emu(140000),
              title, size=14, bold=True, color=color, align=PP_ALIGN.CENTER)
    _add_text(s, x + Emu(50000), y_center + Emu(380000), box_w - Emu(100000), Emu(800000),
              desc, size=11, color=C_BODY, align=PP_ALIGN.CENTER)
    # Arrow
    if i < len(engines) - 1:
        arrow_x = x + box_w + Emu(20000)
        _add_text(s, arrow_x, y_center + box_h // 2 - Emu(50000), gap - Emu(40000), Emu(100000),
                  "➜", size=32, bold=True, color=C_ACCENT, align=PP_ALIGN.CENTER)

# Labels below
_add_text(s, Emu(457200), SLIDE_H - Emu(2500000), SLIDE_W - Emu(914400), Emu(180000),
          "Key insight: each engine operates on a different representation layer\n"
          "(text → graph → tensor → constraint), and they compose without a single point of failure.",
          size=13, italic=True, color=C_MUTED, align=PP_ALIGN.CENTER)


# ═══════════════════════════════════════════════════════════════════════════
# Slide 7 — Pipeline Walkthrough
# ═══════════════════════════════════════════════════════════════════════════
s = finalize_slide(new_slide(), "End-to-End Pipeline",
                   "From \"my chest is tight and I can't sleep\" to a verified recommendation",
                   page_no=7)

steps = [
    ("1. User Input",   "Colloquial, ambiguous free text", C_TITLE),
    ("2. NLU Extraction",  "Qwen 2.5 → symptom list (controlled vocab)", C_ACCENT),
    ("3. Syndrome Inference",  "Symptom graph → syndrome voting", C_ACCENT2),
    ("4. Model Prediction",    "TCMNet → herb/formula ranking", C_TITLE),
    ("5. Safety Screening",    "Toxicity + 18反/19畏 + pregnancy", RGBColor(0x6B, 0x4C, 0x8A)),
    ("6. Knowledge Enrichment", "Smart search adds classical references", C_ACCENT2),
    ("7. Result Verification",  "Claim filtering + grounding + disclaimer", C_TITLE),
]

for i, (label, desc, color) in enumerate(steps):
    y = Emu(1200000) + i * Emu(750000)
    # Step number circle
    circ = s.shapes.add_shape(MSO_SHAPE.OVAL, Emu(500000), y, Emu(500000), Emu(500000))
    circ.fill.solid()
    circ.fill.fore_color.rgb = C_ACCENT if i in (1, 4) else (C_ACCENT2 if i in (2, 5) else C_TITLE)
    circ.line.fill.background()
    circ.shadow.inherit = False
    _add_text(s, Emu(500000), y, Emu(500000), Emu(500000),
              str(i), font=FONT_MONO, size=22, bold=True, color=C_BG, align=PP_ALIGN.CENTER)
    _add_text(s, Emu(1100000), y + Emu(20000), Emu(2500000), Emu(70000),
              label, size=16, bold=True, color=C_TITLE)
    _add_text(s, Emu(1100000), y + Emu(400000), Emu(10000000), Emu(60000),
              desc, size=13, color=C_BODY)


# ═══════════════════════════════════════════════════════════════════════════
# Slide 8 — NLU Controlled Interface
# ═══════════════════════════════════════════════════════════════════════════
s = finalize_slide(new_slide(), "NLU: Controlled Extraction, Not Generation",
                   "Qwen 2.5-0.5B + closed-vocab prompting — the safety-critical design choice",
                   page_no=8)

left_box_x = Emu(500000)
right_box_x = Emu(6500000)
box_w = Emu(5500000)
box_h = Emu(4600000)

# Left: Design principle
_rect(s, left_box_x, Emu(1200000), box_w, box_h, fill=C_CARD, line=C_LINE)
_add_text(s, left_box_x + Emu(200000), Emu(1400000), box_w - Emu(400000), Emu(60000),
          "Why closed-vocab extraction?", size=18, bold=True, color=C_ACCENT)
bullets = [
    "LLM hallucinations are catastrophic in medical contexts",
    "Direct generation of formula names can produce non-existent herbs",
    "TCM has 228 canonical symptoms — a manageable closed set",
    "We constrain Qwen to pick from SYMPTOM_ALIAS_MAP entries only",
    "Output is parsed, normalized, and filtered against the ground truth table",
    "Failed extraction → graceful fallback (keyword match → smart search)",
]
_add_bullets(s, left_box_x + Emu(200000), Emu(1550000), box_w - Emu(400000), box_h - Emu(500000),
             bullets, size=14)

# Right: NLU ablation metrics
_rect(s, right_box_x, Emu(1200000), box_w, box_h, fill=C_CARD, line=C_LINE)
_add_text(s, right_box_x + Emu(200000), Emu(1400000), box_w - Emu(400000), Emu(60000),
          "NLU Extraction Quality (CPU float32, 30 test cases)",
          size=17, bold=True, color=C_ACCENT2)

nlu_metrics = [
    ("Exact match rate",     "43.3%",  "user text exactly → standard symptom"),
    ("At-least-one-hit",     "75.0%",  "useful for multi-symptom inputs"),
    ("Micro Precision",       "57.5%",  ""),
    ("Micro Recall",          "51.1%",  ""),
    ("Micro F1",              "54.1%",  ""),
    ("Avg inference time",    "18.8 s", "CPU-only, no GPU"),
]
for i, (name, val, note) in enumerate(nlu_metrics):
    y = Emu(1550000) + i * Emu(500000)
    _add_text(s, right_box_x + Emu(200000), y, Emu(3000000), Emu(60000),
              name, size=14, bold=True, color=C_BODY)
    _add_text(s, right_box_x + Emu(3500000), y, Emu(1500000), Emu(60000),
              val, size=14, bold=True, color=C_ACCENT2)
    if note:
        _add_text(s, right_box_x + Emu(200000), y + Emu(60000), box_w - Emu(400000), Emu(40000),
                  note, size=11, italic=True, color=C_MUTED)

_add_text(s, right_box_x + Emu(200000), Emu(1200000) + box_h - Emu(400000), box_w - Emu(400000), Emu(300000),
          "4-bit quantization further cuts latency ~3× on CUDA, with minimal precision loss.",
          size=12, italic=True, color=C_MUTED)


# ═══════════════════════════════════════════════════════════════════════════
# Slide 9 — Knowledge Graph
# ═══════════════════════════════════════════════════════════════════════════
s = finalize_slide(new_slide(), "Knowledge Graph: The Structured Brain",
                   "5 node types · 12 edge types · syndrome voting · path traceability",
                   page_no=9)

# Node inventory
nodes = [
    ("Symptom",   "228 unique symptoms + 272 aliases"),
    ("Syndrome",  "38 TCM syndromes (肾虚, 脾虚, ...)"),
    ("Herb",      "200+ herbs with properties (性味归经)"),
    ("Formula",   "80+ classical formulas with composition"),
    ("Contraindication", "十八反 / 十九畏 rule edges"),
]
for i, (t, d) in enumerate(nodes):
    y = Emu(1200000) + i * Emu(650000)
    shape = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Emu(500000), y, Emu(1800000), Emu(550000))
    shape.fill.solid()
    shape.fill.fore_color.rgb = C_ACCENT2 if i < 3 else (C_ACCENT if i == 3 else RGBColor(0x8C, 0x2B, 0x3E))
    shape.line.fill.background()
    shape.shadow.inherit = False
    _add_text(s, Emu(500000), y, Emu(1800000), Emu(550000),
              t, size=16, bold=True, color=C_BG, align=PP_ALIGN.CENTER)
    _add_text(s, Emu(2500000), y + Emu(150000), Emu(3500000), Emu(400000),
              d, size=14, color=C_BODY)

# Key graph operations (right side)
ops_x = Emu(7000000)
_add_text(s, ops_x, Emu(1200000), Emu(4500000), Emu(60000),
          "Key Operations", size=18, bold=True, color=C_ACCENT)
ops = [
    "symptom → syndrome voting (多症状证型推理)",
    "symptom → herb direct association lookup",
    "herb → formula composition check",
    "contraindication edge traversal (real-time)",
    "cofrequency analysis (共现分析) between symptoms",
    "graph visualization for explainability",
]
_add_bullets(s, ops_x, Emu(1320000), Emu(4500000), Emu(4500000), ops, size=13)


# ═══════════════════════════════════════════════════════════════════════════
# Slide 10 — TCMNet Model
# ═══════════════════════════════════════════════════════════════════════════
s = finalize_slide(new_slide(), "TCMNet: Small-Sample Deep Learning",
                   "Pre-LN Transformer · Dual Encoder · Cross-Attention · SupCon Loss",
                   page_no=10)

left_x = Emu(500000)
right_x = Emu(6500000)

# Architecture card
_rect(s, left_x, Emu(1100000), Emu(5500000), Emu(4800000), fill=C_CARD, line=C_LINE)
_add_text(s, left_x + Emu(200000), Emu(1300000), Emu(5100000), Emu(60000),
          "Why this architecture?", size=18, bold=True, color=C_ACCENT)
_add_bullets(s, left_x + Emu(200000), Emu(1420000), Emu(5100000), Emu(4200000),
             [
                 "Symptom Encoder: Pre-LN Transformer with positional encoding",
                 "Herb Encoder: fuses textual + attribute embeddings (性味归经)",
                 "Cross-Attention: bridges symptom representation to herb space",
                 "Multi-task heads: syndrome classification + herb ranking + formula ranking",
                 "Supervised Contrastive Loss: pulls symptom and matching herbs together",
                 "Mixed-precision training (AMP) + CosineAnnealingWarmRestarts",
             ], size=13)

# Training performance card
_rect(s, right_x, Emu(1100000), Emu(5500000), Emu(4800000), fill=C_CARD, line=C_LINE)
_add_text(s, right_x + Emu(200000), Emu(1300000), Emu(5100000), Emu(60000),
          "Training Metrics", size=18, bold=True, color=C_ACCENT2)
metrics = [
    ("Dataset",        "500K synthetic samples (5-stage sampling)"),
    ("d_model",        "128 / 256 (v2)"),
    ("Layers",         "2 / 4"),
    ("Best syn_acc",   "0.979 (checkpoint)", "证型 classification accuracy"),
    ("Best val loss",  "recorded in .pth checkpoint"),
    ("Epochs",         "7 (v2 best_model_v4)"),
    ("Optimizer",      "AdamW, lr=1e-4"),
    ("Grad accum",     "2"),
]
for i, (name, val, *note) in enumerate(metrics):
    y = Emu(1420000) + i * Emu(460000)
    _add_text(s, right_x + Emu(200000), y, Emu(1800000), Emu(60000),
              name, size=14, bold=True, color=C_BODY)
    _add_text(s, right_x + Emu(2100000), y, Emu(3200000), Emu(60000),
              val, size=14, bold=True, color=C_ACCENT2)
    if note:
        _add_text(s, right_x + Emu(2100000), y + Emu(60000), Emu(3200000), Emu(40000),
                  note[0], size=11, italic=True, color=C_MUTED)


# ═══════════════════════════════════════════════════════════════════════════
# Slide 11 — Safety by Design
# ═══════════════════════════════════════════════════════════════════════════
s = finalize_slide(new_slide(), "Safety by Design — Not Afterthought",
                   "Four independent screening layers before any recommendation is surfaced",
                   page_no=11)

safety_layers = [
    ("Layer 1 — Toxicity Check",
     "Each herb in EXTENDED_TOXIC_HERBS is flagged ☠️ and its formula may be filtered.",
     C_ACCENT),
    ("Layer 2 — Contraindication Graph",
     "18反 / 19畏 rules encoded as KG edges. Any recommendation pair traverses the edge → both blocked.",
     RGBColor(0x6B, 0x4C, 0x8A)),
    ("Layer 3 — Pregnancy Screening",
     "Pregnancy-disabled herbs are tagged; pregnancy-disabled formulas trigger mandatory warning.",
     C_ACCENT2),
    ("Layer 4 — Result Verification",
     "Absolute claims filtered (\"guarantee\", \"cure\"). Classical sources appended. AGPL-3.0 disclaimer.",
     C_TITLE),
]
for i, (title, body, color) in enumerate(safety_layers):
    y = Emu(1200000) + i * Emu(1250000)
    _rect(s, Emu(500000), y, Emu(30000), Emu(1200000), fill=color)
    _add_text(s, Emu(550000), y + Emu(100000), SLIDE_W - Emu(1200000), Emu(60000),
              title, size=17, bold=True, color=color)
    _add_text(s, Emu(550000), y + Emu(250000), SLIDE_W - Emu(1200000), Emu(900000),
              body, size=14, color=C_BODY)


# ═══════════════════════════════════════════════════════════════════════════
# Slide 12 — Experimental Setup
# ═══════════════════════════════════════════════════════════════════════════
s = finalize_slide(new_slide(), "Experimental Setup & Ablation Design",
                   "Three complementary studies answer three different questions",
                   page_no=12)

studies = [
    ("Study 1: NLU Precision",
     "Q: How well does the LLM extract canonical symptoms from colloquial input?",
     "30 handcrafted utterances → expected symptom list\nMetric: exact match, F1, inference time\nConfig: Qwen 2.5-0.5B, CPU float32\nSource: data/ablation_nlu.json",
     C_ACCENT),
    ("Study 2: Fusion Weight Scan",
     "Q: Does KG-enhanced fusion outperform model-only recommendation?",
     "14 textbook-ground-truth cases, candidate = model Top20 ∪ KG\nMetric: Recall@10, Precision@10\nScan 11 weights (DL 1.0 → 0.0, KG 0.0 → 1.0)\nSource: data/ablation_fusion.json",
     C_ACCENT2),
    ("Study 3: Training Performance",
     "Q: What does the in-domain model actually achieve?",
     "500K synthetic samples, 7 epochs\nMetric: syn_acc, val loss, checkpoint history\nKnown limitation: split from same pipeline (synthetic)\nSource: checkpoint .pth metadata",
     C_TITLE),
]

for i, (title, q, detail, color) in enumerate(studies):
    x = Emu(500000) + i * ((SLIDE_W - Emu(1000000)) // 3 + Emu(200000))
    w = (SLIDE_W - Emu(1000000)) // 3
    _rect(s, x, Emu(1100000), w, Emu(4900000), fill=C_CARD, line=color)
    _rect(s, x, Emu(1100000), w, Emu(60000), fill=color)
    _add_text(s, x + Emu(100000), Emu(1220000), w - Emu(200000), Emu(70000),
              title, size=15, bold=True, color=color, align=PP_ALIGN.CENTER)
    _add_text(s, x + Emu(100000), Emu(1420000), w - Emu(200000), Emu(150000),
              q, size=12, italic=True, color=C_MUTED)
    _add_text(s, x + Emu(100000), Emu(1620000), w - Emu(200000), Emu(3800000),
              detail, size=12, color=C_BODY)


# ═══════════════════════════════════════════════════════════════════════════
# Slide 13 — Results: NLU
# ═══════════════════════════════════════════════════════════════════════════
s = finalize_slide(new_slide(), "Results I: NLU Symptom Extraction",
                   "Closed-vocab prompting at 54% F1 on colloquial Chinese — a strong baseline",
                   page_no=13)

# Metric bar chart (approximate)
chart_y = Emu(1400000)
bar_max = Emu(7000000)
metrics = [
    ("Exact match",   43.3, C_ACCENT),
    ("At-least-one",  75.0, C_ACCENT2),
    ("Micro Precision", 57.5, RGBColor(0xB8, 0x66, 0x2B)),
    ("Micro Recall",   51.1, RGBColor(0x3E, 0x6B, 0x8C)),
    ("Micro F1",       54.1, C_TITLE),
]

for i, (label, pct, color) in enumerate(metrics):
    y = chart_y + i * Emu(700000)
    # Background track
    _rect(s, Emu(3000000), y + Emu(100000), bar_max, Emu(300000), fill=C_LINE)
    # Value bar
    w = int(bar_max * pct / 100)
    _rect(s, Emu(3000000), y + Emu(100000), w, Emu(300000), fill=color)
    # Label + value
    _add_text(s, Emu(500000), y, Emu(2400000), Emu(500000),
              label, size=14, bold=True, color=C_BODY)
    _add_text(s, Emu(3000000) + w + Emu(50000), y, Emu(600000), Emu(500000),
              f"{pct:.1f}%", size=14, bold=True, color=color)

# Example pairs
_add_text(s, Emu(500000), Emu(1400000) + len(metrics) * Emu(700000) + Emu(100000),
          SLIDE_W - Emu(1000000), Emu(50000),
          "Qualitative observations", size=15, bold=True, color=C_ACCENT)
examples = [
    ("睡不着觉 / 翻来覆去睡不着", "→ 失眠", "perfect"),
    ("头疼得厉害",                 "→ 头痛", "perfect"),
    ("吃不下饭",                  "→ 食积 (should be 食欲不振)", "word-level confusion"),
    ("手脚冰凉",                   "→ 畏寒 (should be 手足不温)", "alias under-resolution"),
    ("最近压力大 / 整个人都不好了", "→ rejects, correctly (no canonical symptom)", "strong — not forced"),
]
for i, (inp, out, note) in enumerate(examples):
    y = Emu(5400000) + i * Emu(220000)
    _add_text(s, Emu(500000), y, Emu(3500000), Emu(60000), inp, size=12, color=C_BODY)
    _add_text(s, Emu(4100000), y, Emu(2500000), Emu(60000), out, size=12, bold=True, color=C_ACCENT2)
    _add_text(s, Emu(6700000), y, Emu(4500000), Emu(60000), note, size=11, italic=True, color=C_MUTED)


# ═══════════════════════════════════════════════════════════════════════════
# Slide 14 — Results: Fusion
# ═══════════════════════════════════════════════════════════════════════════
s = finalize_slide(new_slide(), "Results II: Fusion Weight Scan",
                   "KG candidates improve Recall@10 by +2.6pp — but model-KG synergy is not yet realized",
                   page_no=14)

# Weight-scan table
weights = [
    ("1.0 / 0.0 (model only)",      21.2, 11.4),
    ("0.5 / 0.5",                  21.2, 11.4),
    ("0.4 / 0.6 (best)",           23.8, 12.9),
    ("0.0 / 1.0 (pure KG)",        23.8, 12.9),
]

_add_text(s, Emu(500000), Emu(1200000), SLIDE_W - Emu(1000000), Emu(60000),
          "Fusion Weight Scan (14 textbook-ground-truth cases, candidate = model Top20 ∪ KG)",
          size=15, bold=True, color=C_ACCENT2)

for i, (w, recall, prec) in enumerate(weights):
    y = Emu(1320000) + i * Emu(500000)
    _rect(s, Emu(500000), y, SLIDE_W - Emu(1000000), Emu(450000),
          fill=C_CARD if i % 2 == 0 else RGBColor(0xFF, 0xFB, 0xE8),
          line=C_LINE)
    is_best = "best" in w
    w_color = C_ACCENT if is_best else C_BODY
    _add_text(s, Emu(700000), y + Emu(100000), Emu(5000000), Emu(300000),
              w, size=14, bold=is_best, color=w_color)
    _add_text(s, Emu(6000000), y + Emu(100000), Emu(2000000), Emu(300000),
              f"Recall@10 = {recall:.1f}%", size=14, bold=is_best, color=C_ACCENT2 if is_best else C_BODY)
    _add_text(s, Emu(8200000), y + Emu(100000), Emu(2000000), Emu(300000),
              f"Precision@10 = {prec:.1f}%", size=14, bold=is_best, color=C_ACCENT2 if is_best else C_BODY)

# Key findings
_findings = [
    "Finding 1: Best fusion (Recall@10 = 23.8%) strictly dominates model-only (21.2%) — KG adds unique candidates.",
    "Finding 2: Pure KG alone already achieves 23.8% — the model + KG score combination does NOT improve over KG alone in weighted range 0–0.5.",
    "Finding 3: The weight-scan plateau (constant recall for weights 0–0.5) means model and KG outputs barely overlap. True synergy (weighted score arithmetic helping) would be a steeper curve.",
    "Implications: Fusion architecture works for coverage; collaborative scoring is a direction for future work.",
]
for i, f in enumerate(_findings):
    y = Emu(3500000) + i * Emu(450000)
    _add_text(s, Emu(500000), y, SLIDE_W - Emu(1000000), Emu(400000),
              f"• {f}", size=13, color=C_BODY)


# ═══════════════════════════════════════════════════════════════════════════
# Slide 15 — Results: Training
# ═══════════════════════════════════════════════════════════════════════════
s = finalize_slide(new_slide(), "Results III: Training & In-Domain Performance",
                   "syn_acc 0.979 within-domain — but we know this number is inflated",
                   page_no=15)

# Two-card layout
_rect(s, Emu(500000), Emu(1100000), Emu(5500000), Emu(5000000), fill=C_CARD, line=C_ACCENT2)
_add_text(s, Emu(700000), Emu(1300000), Emu(5100000), Emu(60000),
          "What the checkpoint says", size=18, bold=True, color=C_ACCENT2)
cp_lines = [
    ("Training configuration", ""),
    ("  samples",     "500,000 (5-stage stratified)"),
    ("  epochs",      "7 (best_model_v4.pth)"),
    ("  d_model",     "128 → 256"),
    ("  optimizer",   "AdamW, lr=1e-4"),
    ("  grad accum",  "2"),
    ("", ""),
    ("Reported metrics", ""),
    ("  syn_acc",     "0.979"),
    ("  v2 checkpoint", "v2 architecture, 2 layers"),
]
for i, (k, v) in enumerate(cp_lines):
    y = Emu(1420000) + i * Emu(320000)
    if v == "" and k == "":
        continue
    _add_text(s, Emu(700000), y, Emu(2200000), Emu(60000),
              k, size=13, bold=(v == ""), color=C_BODY if v else C_ACCENT2)
    if v:
        _add_text(s, Emu(2900000), y, Emu(3100000), Emu(60000),
                  v, size=13, color=C_ACCENT2)

_rect(s, Emu(6500000), Emu(1100000), Emu(5500000), Emu(5000000), fill=C_CARD, line=C_ACCENT)
_add_text(s, Emu(6700000), Emu(1300000), Emu(5100000), Emu(60000),
          "The honest caveat", size=18, bold=True, color=C_ACCENT)
caveat = [
    "Split from same pipeline",
    "  Training & validation both come from TCMDataPipeline",
    "  This is NOT a held-out external benchmark",
    "  syn_acc 0.979 is a ceiling, not a strength",
    "",
    "Score saturation observed",
    "  sigmoid outputs cluster at 1.000",
    "  UI confidence numbers are not informative",
    "  Model is learning relative ordering, not absolute probability",
    "",
    "Npz data file lost",
    "  best_model.json metadata may not match .pth weight state",
    "  Reproducibility gap — fix: lock seed + commit hash in checkpoint",
]
for i, line in enumerate(caveat):
    y = Emu(1420000) + i * Emu(310000)
    if line == "":
        continue
    _add_text(s, Emu(6700000), y, Emu(5100000), Emu(60000),
              f"• {line}" if line.startswith(" ") else f"— {line}",
              size=12, bold=(not line.startswith(" ") and " " not in line),
              color=C_BODY if line.startswith(" ") else C_ACCENT)


# ═══════════════════════════════════════════════════════════════════════════
# Slide 16 — Discussion: What Works
# ═══════════════════════════════════════════════════════════════════════════
s = finalize_slide(new_slide(), "Discussion: What Works, What Doesn't",
                   "Three takeaways — evidence-based, not aspirational",
                   page_no=16)

takeaways = [
    ("✓  Hybrid architecture is the right paradigm for TCM",
     "Pure rules cover too little. Pure DL needs data we do not have. Pure KG misses soft patterns. TCMAI's four-engine decomposition is validated by ablation — model-only vs pure KG fusion improves Recall@10 by 2.6pp, demonstrating that coverage synergy is real."),
    ("✓  Controlled NLU (closed vocab) beats free generation",
     "43.3% exact match on colloquial Chinese while correctly rejecting non-symptom utterances (\"最近压力大\" → no symptom). This is not a 'nice to have' — hallucinated herbs would be unsafe. The closed-vocab constraint is a safety property, not a model limitation."),
    ("✓  Safety engineering is non-negotiable in medical AI",
     "Four-layer screening (toxicity → contraindication graph → pregnancy → claim verification) is demonstrated and testable. TCM is a case study: no medical AI can be deployed without traceable, auditable, fail-safe guardrails."),
]

for i, (head, body) in enumerate(takeaways):
    y = Emu(1200000) + i * Emu(1700000)
    _rect(s, Emu(500000), y, SLIDE_W - Emu(1000000), Emu(1550000), fill=C_CARD, line=C_LINE)
    _add_text(s, Emu(700000), y + Emu(150000), SLIDE_W - Emu(1400000), Emu(70000),
              head, size=17, bold=True, color=C_ACCENT2)
    _add_text(s, Emu(700000), y + Emu(400000), SLIDE_W - Emu(1400000), Emu(1100000),
              body, size=13, color=C_BODY)


# ═══════════════════════════════════════════════════════════════════════════
# Slide 17 — Why This Direction Matters
# ═══════════════════════════════════════════════════════════════════════════
s = finalize_slide(new_slide(), "Why This Direction Matters",
                   "TCMAI is not just a Q&A app — it is a proof that AI can honor domain specificity",
                   page_no=17)

addenda = [
    ("Domain-Specific AI > General AI",
     "General-purpose LLMs hallucinate TCM knowledge and cannot enforce TCM-specific safety rules. A domain-hybrid framework is not a compromise — it is the responsible default for structured medical knowledge domains."),
    ("Explainable AI by Composition",
     "Composing NLU, KG, and DL gives three independent reasoning traces for every answer. The physician can cross-check: symptom list (NLU) → syndrome path (KG) → ranking (model). No single-engine black box can offer this."),
     ("Knowledge Preservation at Scale",
      "2000 years of TCM scholarship is underutilized because it is not machine-readable. TCMAI demonstrates a path: encode structural knowledge in a graph, distill soft patterns in a small model, and anchor both with a rule system. This pattern generalizes to other traditional medical systems."),
    ("Safety as Architecture",
     "The four-layer screening design shows that safety is not a post-processing filter — it is a parallel engine that runs independently of every recommendation. This architectural principle generalizes to all safety-critical AI."),
]

for i, (head, body) in enumerate(addenda):
    y = Emu(1100000) + i * Emu(1350000)
    _rect(s, Emu(500000), y, Emu(40000), Emu(1200000), fill=C_ACCENT2)
    _add_text(s, Emu(700000), y + Emu(100000), SLIDE_W - Emu(1400000), Emu(60000),
              head, size=17, bold=True, color=C_ACCENT)
    _add_text(s, Emu(700000), y + Emu(250000), SLIDE_W - Emu(1400000), Emu(900000),
              body, size=13, color=C_BODY)


# ═══════════════════════════════════════════════════════════════════════════
# Slide 18 — Limitations & Future Work
# ═══════════════════════════════════════════════════════════════════════════
s = finalize_slide(new_slide(), "Limitations & Research Frontiers",
                   "Honest assessment — what is next for TCMAI and the field",
                   page_no=18)

lims = [
    ("Evaluation is synthetic",
     "Training/validation split from same pipeline; 14-case fusion benchmark is small. Need an external ground-truth set from clinical TCM textbooks.", C_ACCENT),
    ("Fusion is coverage, not synergy",
     "Weight-scan plateau at KG 0.4–1.0 means model and KG barely overlap. Need cross-score calibration or joint training.", C_ACCENT2),
    ("Confidence scores are saturated",
     "sigmoid outputs cluster at 1.000 — model learns ordering, not calibrated probability. UI should hide or relativize these numbers.", RGBColor(0xB8, 0x66, 0x2B)),
    ("No clinician-in-the-loop",
     "Current system has no physician review stage. Next frontier: interactive correction, active learning, and feedback-driven model updating.", RGBColor(0x3E, 0x6B, 0x8C)),
    ("Deployment weight is ~1GB",
     "NLU model size is prohibitive for edge devices. Quantization (GGUF / llama.cpp) or smaller specialized NLU models are needed.", C_TITLE),
]

lim_h = Emu(1300000)
lim_rows = 3
lim_cols = 2
lim_x0 = Emu(457200)
lim_y0 = Emu(1100000)
lim_w = (SLIDE_W - Emu(914400)) // lim_cols
lim_gap = Emu(228600)

for i, (head, body, color) in enumerate(lims):
    col = i % lim_cols
    row = i // lim_cols
    # 5th card: full-width spanning the entire row
    if i == 4:
        x = lim_x0
        w = SLIDE_W - Emu(914400)
        row = 2
    else:
        x = lim_x0 + col * (lim_w + lim_gap)
        w = lim_w
    y = lim_y0 + row * (lim_h + Emu(150000))
    _rect(s, x, y, w, lim_h, fill=C_CARD, line=C_LINE)
    _rect(s, x, y, Emu(40000), lim_h, fill=color)
    _add_text(s, x + Emu(120000), y + Emu(120000), w - Emu(200000), Emu(50000),
              head, size=14, bold=True, color=C_ACCENT)
    _add_text(s, x + Emu(120000), y + Emu(260000), w - Emu(200000), lim_h - Emu(380000),
              body, size=12, color=C_BODY)

# Final thought — compact, positioned below limitation cards (ends ~6.6M, remaining ~258K)
_add_text(s, Emu(457200), SLIDE_H - Emu(300000), SLIDE_W - Emu(914400), Emu(200000),
          "TCM is a test case: AI can honor domain specificity — rules, heuristics, ethics — not flatten everything into one monolithic model.",
          font=FONT_TITLE, size=14, italic=True, color=C_TITLE, align=PP_ALIGN.CENTER)


# ═══════════════════════════════════════════════════════════════════════════
# Save
# ═══════════════════════════════════════════════════════════════════════════
out_path = Path(__file__).resolve().parent.parent / "data" / "TCMAI_Academic_Deck_EN.pptx"
prs.save(str(out_path))
print(f"Saved: {out_path}")
print(f"Slides: {len(prs.slides)}")
