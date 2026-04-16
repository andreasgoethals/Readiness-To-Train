"""
Generate Project Overview PDF

Static problem statement for the Readiness-to-Train causal ML project.
Covers: what we have, what we want, why it's causal, temporal semantics,
sports science foundation. Does NOT cover methods, progress, or results.

Output: Project Overview.pdf (project root)

Usage:
    python src/utils/generate_project_overview.py
"""

from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────────────────────
NAVY   = colors.HexColor('#1B3A6B')   # headings
STEEL  = colors.HexColor('#2E6DA4')   # sub-headings / accents
AMBER  = colors.HexColor('#D4860A')   # callout boxes
LIGHT  = colors.HexColor('#EBF3FB')   # table header fill
PALE   = colors.HexColor('#F7FBFF')   # alternating row fill
GREY   = colors.HexColor('#555555')   # body text
LTGREY = colors.HexColor('#DDDDDD')   # horizontal rules / borders

PAGE_W, PAGE_H = A4
MARGIN = 2.0 * cm
CONTENT_W = PAGE_W - 2 * MARGIN


# ─────────────────────────────────────────────────────────────────────────────
# Style sheet
# ─────────────────────────────────────────────────────────────────────────────
def _styles():
    styles = {}

    styles['title'] = ParagraphStyle(
        'title',
        fontName='Helvetica-Bold',
        fontSize=22,
        textColor=NAVY,
        spaceBefore=0,
        spaceAfter=16,    # enough room so subtitle doesn't overlap
        leading=28,
        alignment=TA_CENTER,
    )
    styles['subtitle'] = ParagraphStyle(
        'subtitle',
        fontName='Helvetica',
        fontSize=12,
        textColor=STEEL,
        spaceBefore=6,
        spaceAfter=4,
        leading=16,
        alignment=TA_CENTER,
    )
    styles['meta'] = ParagraphStyle(
        'meta',
        fontName='Helvetica-Oblique',
        fontSize=9,
        textColor=GREY,
        spaceAfter=2,
        alignment=TA_CENTER,
    )
    styles['h1'] = ParagraphStyle(
        'h1',
        fontName='Helvetica-Bold',
        fontSize=13,
        textColor=NAVY,
        spaceBefore=14,
        spaceAfter=4,
    )
    styles['h2'] = ParagraphStyle(
        'h2',
        fontName='Helvetica-Bold',
        fontSize=11,
        textColor=STEEL,
        spaceBefore=10,
        spaceAfter=3,
    )
    styles['body'] = ParagraphStyle(
        'body',
        fontName='Helvetica',
        fontSize=9.5,
        textColor=GREY,
        leading=14,
        spaceAfter=6,
        alignment=TA_JUSTIFY,
    )
    styles['bullet'] = ParagraphStyle(
        'bullet',
        fontName='Helvetica',
        fontSize=9.5,
        textColor=GREY,
        leading=14,
        leftIndent=12,
        spaceAfter=4,
        bulletIndent=0,
    )
    styles['callout'] = ParagraphStyle(
        'callout',
        fontName='Helvetica-BoldOblique',
        fontSize=9.5,
        textColor=AMBER,
        leading=14,
        leftIndent=10,
        rightIndent=10,
        spaceAfter=4,
        alignment=TA_JUSTIFY,
    )
    styles['th'] = ParagraphStyle(
        'th',
        fontName='Helvetica-Bold',
        fontSize=8.5,
        textColor=NAVY,
        alignment=TA_CENTER,
    )
    styles['td'] = ParagraphStyle(
        'td',
        fontName='Helvetica',
        fontSize=8.5,
        textColor=GREY,
        leading=12,
        alignment=TA_LEFT,
    )
    styles['tdc'] = ParagraphStyle(
        'tdc',
        fontName='Helvetica',
        fontSize=8.5,
        textColor=GREY,
        leading=12,
        alignment=TA_CENTER,
    )
    return styles


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _hr(story):
    story.append(HRFlowable(width='100%', thickness=0.5, color=LTGREY,
                             spaceAfter=6, spaceBefore=2))


def _section(story, title, styles):
    story.append(Paragraph(title, styles['h1']))
    _hr(story)


def _table(story, header, rows, styles, col_widths=None):
    """Build a styled table with alternating row shading."""
    data = [[Paragraph(c, styles['th']) for c in header]]
    for row in rows:
        data.append([Paragraph(str(c), styles['td']) for c in row])

    t = Table(data, colWidths=col_widths, repeatRows=1,
              hAlign='CENTER', splitByRow=True)

    style_cmds = [
        ('BACKGROUND', (0, 0), (-1, 0), LIGHT),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, PALE]),
        ('GRID', (0, 0), (-1, -1), 0.4, LTGREY),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]
    t.setStyle(TableStyle(style_cmds))
    story.append(t)
    story.append(Spacer(1, 8))


def _callout(story, text, styles):
    story.append(Paragraph(f'&#9888; {text}', styles['callout']))


def _page_number(canvas, doc):
    """Draw page number at bottom centre of each page."""
    canvas.saveState()
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(GREY)
    canvas.drawCentredString(PAGE_W / 2, 1.2 * cm, str(doc.page))
    canvas.restoreState()


# ─────────────────────────────────────────────────────────────────────────────
# PDF content
# ─────────────────────────────────────────────────────────────────────────────
def build_pdf(output_path: Path):
    from reportlab.platypus import Image
    S = _styles()
    story = []
    project_root = output_path.parent

    # ── Cover ─────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 1.8 * cm))
    story.append(Paragraph('Readiness to Train', S['title']))
    story.append(Paragraph(
        'Prescriptive Analytics for Optimal Training Intensity', S['subtitle']))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph(
        'KU Leuven &amp; OH Leuven', S['meta']))
    story.append(Paragraph('Project Overview &amp; Problem Statement', S['meta']))
    story.append(Spacer(1, 0.8 * cm))
    _hr(story)
    story.append(Spacer(1, 0.4 * cm))

    # ── 1. Core Objective ─────────────────────────────────────────────────────
    _section(story, '1. Core Objective', S)

    story.append(Paragraph(
        'The goal of this project is to <b>optimise the performance of players on '
        'match day</b> by developing a prescriptive analytics system that estimates '
        'individual player <b>Readiness to Train</b>. Using <b>Causal Machine '
        'Learning</b> within a <b>Dynamic Treatment Regime (DTR)</b> framework, the '
        'system will recommend an optimal daily training intensity for each player '
        '&#8212; delivered as a continuous score between 0 and 1. A score of 0 '
        'indicates the player should rest; a score of 1 indicates the player can '
        'train at maximal intensity.',
        S['body']))

    story.append(Paragraph(
        'The system will be fully <b>individualised</b>: each player will receive a '
        'personalised score derived from their own physiological profile, training '
        'history, and evolving daily state. The ultimate objective is to find the '
        'training intensity sequence across the pre-match cycle that maximises each '
        'player\'s physical output on match day.',
        S['body']))

    # ── 2. Partnership & Data ─────────────────────────────────────────────────
    _section(story, '2. Partnership & Data', S)

    story.append(Paragraph(
        'The project is a collaboration between <b>KU Leuven</b> (methodology, '
        'analysis) and <b>OH Leuven</b> (professional football club, data provider). '
        'OH Leuven operates a systematic daily player monitoring programme covering '
        'GPS load metrics, subjective wellness questionnaires, medical status, and '
        'match performance. Four raw datasets are available:',
        S['body']))

    _table(story,
           ['Dataset', 'Rows', 'Cols', 'Players', 'Start', 'End', 'Days'],
           [
               ['Readiness_Data', '14,359', '24', '28',
                '2024-07-01', '2026-02-17', '597'],
               ['Raw_Data', '9,968', '39', '84',
                '2024-05-02', '2026-03-01', '670'],
               ['Sessions', '1,206', '9', '&#8212;',
                '2024-05-02', '2026-03-01', '670'],
               ['Games', '403', '9', '24',
                '2025-07-27', '2026-02-28', '216'],
           ], S,
           col_widths=[4.0*cm, 1.8*cm, 1.5*cm, 2.0*cm, 3.0*cm, 3.0*cm, 1.7*cm])

    story.append(Paragraph(
        'The <b>processed dataset</b> (RTT.xlsx) merges all four sources into a '
        'single analysis-ready file with 46 engineered features per player-day. '
        'Readiness_Data is the base; Raw_Data provides detailed GPS/HR at session '
        'level (shifted +1 day so it appears as "yesterday" data); Games provides '
        'match performance data (also shifted +1 day). Player overlap: all 28 '
        'Readiness_Data players appear in Raw_Data; 23 of 28 appear in Games.',
        S['body']))

    story.append(Paragraph(
        '<b>Data dictionaries:</b> Two auto-generated PDF documents describe '
        'every variable in detail. The <b>Raw Data Dictionary</b> '
        '(<i>data/raw/Raw Data Dictionary.pdf</i>) documents all columns across '
        'the four raw datasets. The <b>RTT Data Dictionary</b> '
        '(<i>data/processed/RTT Data Dictionary.pdf</i>) documents all 46 columns '
        'in the processed dataset, including engineered features and their temporal '
        'positions.',
        S['body']))

    story.append(Paragraph(
        '<b>Temporal ordering within each row:</b> Variables in a single row have '
        'different temporal positions. "Yesterday" columns contain data from day '
        't&#8722;1 (fully known). Morning wellness z-scores are measured at the '
        'start of day t (before any activity decisions). Post-assessment variables '
        '(Activity Type Today, Selected) are determined after the morning assessment '
        'and represent the treatment decision &#8212; they must never be used as '
        'predictors in standard ML.',
        S['body']))

    # ── 3. Sports Science Foundation ──────────────────────────────────────────
    _section(story, '3. Sports Science Foundation', S)

    story.append(Paragraph(
        'Training stress produces fatigue, which during recovery is followed by '
        'adaptation (supercompensation). The ACWR (Acute:Chronic Workload Ratio) '
        'captures where a player sits on this curve: the sweet spot is '
        '0.8&#8211;1.3; values above 1.5 are flagged as danger zones (Gabbett, '
        '2016). GPS metrics are individualised as percentages of each player\'s '
        'match benchmarks. The training intensity score A(t) is the '
        'tanh-compressed harmonic mean of these GPS percentages, soft-capped '
        'in [0,&nbsp;1). The typical weekly structure (5&#8211;6 training days '
        'followed by a match) creates player-specific match cycles central to '
        'the DTR formulation.',
        S['body']))

    # ── 4. Why This Is a Causal Problem ──────────────────────────────────────
    _section(story, '4. Why This Is a Causal Problem', S)

    story.append(Paragraph(
        'Standard predictive modelling asks <i>"given the data, what will '
        'happen?"</i>. This project asks <i>"given a player\'s full history up '
        'to today, what training intensity today best contributes to future '
        'match-day performance?"</i>. This requires causal reasoning for three '
        'reasons:',
        S['body']))

    story.append(Paragraph(
        '<b>1. Time-Varying Confounding Affected by Prior Treatment.</b> '
        'Covariates at time t (fatigue, readiness, ACWR) are simultaneously '
        '<i>consequences</i> of treatment at t&#8722;1 and <i>causes</i> of '
        'treatment at t. Classical regression cannot handle this correctly &#8212; '
        'this is precisely the setting for which G-methods were developed.',
        S['bullet']))

    story.append(Paragraph(
        '<b>2. Treatment-Confounder Feedback.</b> Coaching decisions are '
        'observational. Hard sessions are prescribed when players appear fresh; '
        'recovery is prescribed when fatigue is high. The observed '
        'training-performance association conflates physiological response with '
        'coach selection behaviour.',
        S['bullet']))

    story.append(Paragraph(
        '<b>3. Sequential Decision-Making.</b> The treatment is a repeated, '
        'time-varying intervention. The question is not just "what training '
        'today?" but "what <i>sequence</i> of training intensities over the '
        'cycle maximises match-day performance?" &#8212; a DTR problem.',
        S['bullet']))

    story.append(Spacer(1, 4))

    # ── 5. Causal Framework ─────────────────────────────────────────────────
    _section(story, '5. Causal Framework', S)

    _table(story,
           ['Symbol', 'Role', 'Description'],
           [
               ['L(t)', 'Covariates',
                'Player state at time t: wellness, ACWR, GPS load, schedule position'],
               ['A(t)', 'Treatment',
                'Training intensity [0,1]: tanh(harmonic mean of GPS benchmark %)'],
               ['Y', 'Outcome',
                'Match-day performance: high-intensity metrics per ball-in-play minute'],
           ], S,
           col_widths=[1.8*cm, 2.5*cm, 11.9*cm])

    story.append(Paragraph(
        'The problem is modelled as a <b>longitudinal causal DAG</b> (Directed '
        'Acyclic Graph) with the following structure within each match cycle:',
        S['body']))

    story.append(Paragraph(
        '&#8226; At each time step t, the player\'s <b>covariates L(t)</b> '
        '(morning wellness, fatigue, ACWR) causally affect the coaching '
        'staff\'s <b>treatment decision A(t)</b> (what training intensity is '
        'prescribed that afternoon).',
        S['bullet']))

    story.append(Paragraph(
        '&#8226; The treatment A(t) then causally affects the player\'s '
        '<b>state L(t+1)</b> the next morning (training changes fatigue, '
        'soreness, adaptation level).',
        S['bullet']))

    story.append(Paragraph(
        '&#8226; This creates a <b>feedback loop</b>: L(t) &#8594; A(t) '
        '&#8594; L(t+1) &#8594; A(t+1) &#8594; ... repeating across the '
        'match cycle until match day, when the final player state and '
        'cumulative training determine the <b>match outcome Y</b>.',
        S['bullet']))

    story.append(Paragraph(
        '&#8226; Between cycles, the match outcome Y feeds back into the '
        'first state of the next cycle (match fatigue carries over).',
        S['bullet']))

    story.append(Paragraph(
        'The figure below shows an example DAG for Player 1, match cycle 4 '
        '(schematic view with grouped covariates):',
        S['body']))

    # Insert DAG image -- preserve original aspect ratio
    dag_img = project_root / 'images' / 'DAGs' / 'player 1' / 'player_1_cycle_4_schematic.png'
    if dag_img.exists():
        # Read actual image dimensions to preserve aspect ratio
        try:
            from PIL import Image as PILImage
            pil_img = PILImage.open(str(dag_img))
            img_w, img_h = pil_img.size
            aspect = img_w / img_h
            display_w = CONTENT_W
            display_h = display_w / aspect
        except Exception:
            display_w = CONTENT_W
            display_h = CONTENT_W * 0.6
        story.append(Spacer(1, 4))
        story.append(Image(str(dag_img), width=display_w, height=display_h))
        story.append(Spacer(1, 4))
        story.append(Paragraph(
            '<i>Figure: Causal DAG for Player 1, match cycle 4 (schematic). '
            'Blue ellipses = player state L(t), amber rectangles = treatment A(t), '
            'crimson diamond = match outcome Y.</i>', S['meta']))
    else:
        story.append(Paragraph(
            '<i>[DAG image not found &#8212; run src/utils/generate_visualizations.py]</i>',
            S['meta']))

    story.append(Paragraph(
        'This DAG structure is generated automatically for each player and '
        'each match cycle using the DAGCreator class (499 DAGs total, stored '
        'in <i>images/DAGs/player X/</i>). It encodes the causal assumptions '
        'needed for G-methods: confounding edges (L &#8594; A), treatment '
        'effects (A &#8594; L at t+1), state carryover (L &#8594; L at t+1), '
        'and inter-cycle feedback (Y &#8594; L at cycle k+1).',
        S['body']))

    # ── 6. Key Challenges ─────────────────────────────────────────────────────
    _section(story, '6. Key Challenges', S)

    story.append(Paragraph('<b>Small N, Large T</b>', S['h2']))
    story.append(Paragraph(
        'Only 28 players are monitored in Readiness_Data, but each player has '
        'up to 597 days of observations. The signal is likely dominated by '
        '<b>within-player dynamics</b>. Models must be parsimonious or leverage '
        'partial pooling to avoid overfitting across players. Methods must '
        'balance individual-level tailoring with the limited between-player '
        'sample size.',
        S['body']))

    story.append(Paragraph('<b>Time-Varying Confounding Affected by Prior Treatment</b>', S['h2']))
    story.append(Paragraph(
        'Covariates L(t) at time t are simultaneously consequences of treatment '
        'at t&#8722;1 and causes of treatment at t. This is the defining feature of '
        'the problem and the reason standard regression fails. The appropriate '
        'causal identification framework is the theory of G-methods (Robins, '
        '1986): G-computation, Marginal Structural Models with IPTW, or '
        'G-Estimation / Structural Nested Mean Models.',
        S['body']))

    story.append(Paragraph('<b>Observational Data with Coach Selection Bias</b>', S['h2']))
    story.append(Paragraph(
        'Training loads are prescribed by coaching staff based on observed '
        'player state. Players who appear fresh receive harder sessions; fatigued '
        'players receive recovery sessions. This creates a systematic association '
        'between baseline health and training dose that confounds any naive '
        'regression of load on performance.',
        S['body']))

    story.append(Paragraph('<b>Sparse Match-Day Outcomes</b>', S['h2']))
    story.append(Paragraph(
        'The outcome Y (match performance) is observed only on match days. '
        'With a 1-week cycle, each player contributes approximately one outcome '
        'observation per 6 rows of covariate/treatment data. The Games dataset '
        'covers 24 players over 27 match dates (403 player-match rows). Mean '
        'minutes played: 67.4 min/match; mean high-intensity distance per BIP: '
        '18.75 m/min.',
        S['body']))

    # Modelling approach removed -- covered in Project Results.pdf

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.5 * cm))
    _hr(story)
    story.append(Paragraph(
        'KU Leuven &amp; OH Leuven &#8212; Causal ML Project',
        S['meta']))

    # ── Build ─────────────────────────────────────────────────────────────────
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=2.0 * cm,   # extra room for page numbers
        title='Readiness to Train \u2014 Project Overview',
        author='KU Leuven & OH Leuven',
        subject='Prescriptive Analytics for Optimal Training Intensity',
    )
    doc.build(story, onFirstPage=_page_number, onLaterPages=_page_number)
    print(f'Saved: {output_path}')


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    project_root = Path(__file__).parent.parent.parent
    output_path = project_root / 'Project Overview.pdf'
    build_pdf(output_path)
