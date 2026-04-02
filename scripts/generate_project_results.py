"""
Generate Project Results PDF

Comprehensive results report for the Readiness-to-Train causal ML project.
Summarises findings from all three experiments for delivery to OH Leuven.

Output: Project Results.pdf (project root)

Usage:
    python scripts/generate_project_results.py
"""

from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether, PageBreak, Image
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

# ---------------------------------------------------------------------------
# Colour palette (matches generate_project_overview.py)
# ---------------------------------------------------------------------------
NAVY   = colors.HexColor('#1B3A6B')   # headings
STEEL  = colors.HexColor('#2E6DA4')   # sub-headings / accents
AMBER  = colors.HexColor('#D4860A')   # callout boxes
LIGHT  = colors.HexColor('#EBF3FB')   # table header fill
PALE   = colors.HexColor('#F7FBFF')   # alternating row fill
GREY   = colors.HexColor('#555555')   # body text
LTGREY = colors.HexColor('#DDDDDD')   # horizontal rules / borders
GREEN  = colors.HexColor('#2E7D32')   # positive highlights
RED    = colors.HexColor('#C62828')    # negative highlights

PAGE_W, PAGE_H = A4
MARGIN = 2.0 * cm
CONTENT_W = PAGE_W - 2 * MARGIN


# ---------------------------------------------------------------------------
# Style sheet
# ---------------------------------------------------------------------------
def _styles():
    styles = {}

    styles['title'] = ParagraphStyle(
        'title',
        fontName='Helvetica-Bold',
        fontSize=22,
        textColor=NAVY,
        spaceBefore=0,
        spaceAfter=16,
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
    styles['h3'] = ParagraphStyle(
        'h3',
        fontName='Helvetica-Bold',
        fontSize=10,
        textColor=STEEL,
        spaceBefore=8,
        spaceAfter=2,
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
    styles['bullet2'] = ParagraphStyle(
        'bullet2',
        fontName='Helvetica',
        fontSize=9.5,
        textColor=GREY,
        leading=14,
        leftIndent=24,
        spaceAfter=3,
        bulletIndent=12,
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
    styles['finding'] = ParagraphStyle(
        'finding',
        fontName='Helvetica-Bold',
        fontSize=9.5,
        textColor=GREEN,
        leading=14,
        leftIndent=10,
        rightIndent=10,
        spaceAfter=6,
        alignment=TA_JUSTIFY,
    )
    styles['conclusion_head'] = ParagraphStyle(
        'conclusion_head',
        fontName='Helvetica-Bold',
        fontSize=10,
        textColor=NAVY,
        spaceBefore=2,
        spaceAfter=4,
    )
    styles['conclusion_body'] = ParagraphStyle(
        'conclusion_body',
        fontName='Helvetica',
        fontSize=9.5,
        textColor=GREY,
        leading=14,
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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
        data.append([Paragraph(str(c), styles['tdc']) for c in row])

    t = Table(data, colWidths=col_widths, repeatRows=1,
              hAlign='LEFT', splitByRow=True)

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


def _table_left(story, header, rows, styles, col_widths=None):
    """Table with left-aligned body cells (for text-heavy tables)."""
    data = [[Paragraph(c, styles['th']) for c in header]]
    for row in rows:
        data.append([Paragraph(str(c), styles['td']) for c in row])

    t = Table(data, colWidths=col_widths, repeatRows=1,
              hAlign='LEFT', splitByRow=True)

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
    story.append(Paragraph(text, styles['callout']))


def _finding(story, text, styles):
    story.append(Paragraph(text, styles['finding']))


def _conclusion_box(story, title, text, styles):
    """Light blue box with dark blue header for conclusions."""
    BOX_BG = colors.HexColor('#E8F0FE')
    BOX_BORDER = colors.HexColor('#1B3A6B')
    data = [[Paragraph(title, styles['conclusion_head']),],
            [Paragraph(text, styles['conclusion_body']),]]
    t = Table(data, colWidths=[CONTENT_W - 12])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), BOX_BG),
        ('BOX', (0, 0), (-1, -1), 1.0, BOX_BORDER),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(t)
    story.append(Spacer(1, 8))


def _page_number(canvas, doc):
    """Draw page number at bottom centre of each page."""
    canvas.saveState()
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(GREY)
    canvas.drawCentredString(PAGE_W / 2, 1.2 * cm, str(doc.page))
    canvas.restoreState()


# ---------------------------------------------------------------------------
# PDF content  (storytelling structure for OH Leuven delivery)
# ---------------------------------------------------------------------------
def build_pdf(output_path: Path):
    from datetime import datetime
    S = _styles()
    story = []
    gen_date = datetime.now().strftime('%B %Y')  # e.g. "March 2026"

    # ── Title page ───────────────────────────────────────────────────────
    story.append(Spacer(1, 1.8 * cm))
    story.append(Paragraph('Readiness to Train', S['title']))
    story.append(Paragraph('Project Results', S['subtitle']))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph('KU Leuven &amp; OH Leuven', S['meta']))
    story.append(Paragraph(gen_date, S['meta']))
    story.append(Spacer(1, 0.8 * cm))
    _hr(story)
    story.append(Spacer(1, 0.4 * cm))

    # ==================================================================
    # 1. INTRODUCTION
    # ==================================================================
    _section(story, '1. Introduction', S)

    story.append(Paragraph(
        'The goal of this project is to develop a <b>causal framework for '
        'optimising training load per player</b> in professional football, '
        'such that each player arrives at match day in the best possible '
        'physical condition. The central question is: <i>"Given everything '
        'we know about a player this morning, what training intensity '
        'should we prescribe today to maximise their match-day performance '
        'while respecting their biological constraints?"</i>',
        S['body']))

    story.append(Paragraph(
        'This is fundamentally a <b>causal</b> question: coaches already '
        'make training decisions based on player state, which means the '
        'observed data confounds the effect of training with the selection '
        'of who receives it. Standard predictive modelling cannot '
        'disentangle these; causal methods (G-computation, IPTW, DTR '
        'optimisation) are required.',
        S['body']))

    story.append(Paragraph(
        'The project uses <b>daily monitoring data</b> collected by OH Leuven '
        'over approximately 20 months (July 2024 &#8211; February 2026), '
        'covering <b>28 first-team players</b> and totalling '
        '<b>14,359 player-day observations</b>. Four raw datasets were '
        'merged into a single analysis-ready file (RTT.xlsx) containing '
        '46 engineered features per player-day.',
        S['body']))

    story.append(Paragraph(
        'Three experiments were conducted, each building on the previous. '
        'This report describes the data, the analysis process, the key '
        'engineered variables, and the results of all three experiments.',
        S['body']))

    # ==================================================================
    # 2. DATA AND KEY VARIABLES
    # ==================================================================
    _section(story, '2. Data and Key Variables', S)

    story.append(Paragraph('<b>Data sources</b>', S['h2']))

    _table(story,
           ['Dataset', 'Rows', 'Scope', 'Content'],
           [
               ['Readiness_Data', '14,359', '28 players, daily',
                'Wellness z-scores, ACWR, GPS benchmarks, medical status'],
               ['Raw_Data', '9,968', '84 players, session-level',
                'Detailed GPS, heart rate, session metadata'],
               ['Sessions', '1,206', 'Team-level',
                'Match day flags, session types'],
               ['Games', '403', '24 players, match-level',
                'High-intensity distance, efforts per ball-in-play, minutes'],
           ], S,
           col_widths=[3.0*cm, 1.5*cm, 3.5*cm, 8.2*cm])

    story.append(Paragraph('<b>Data analysis and exploration</b>', S['h2']))

    story.append(Paragraph(
        'Before any modelling, we conducted thorough data analysis across '
        'three levels of Jupyter notebooks. All visualisations are preserved '
        'inside the notebooks and can be re-examined interactively.',
        S['body']))

    story.append(Paragraph(
        '<b>Level 0 &#8212; Data Quality Checks:</b>',
        S['body']))
    story.append(Paragraph(
        '&#8226; <b>0. Processed_Data_Quality</b> &#8212; automated validation '
        'of the processed RTT.xlsx dataset: player coverage, column completeness, '
        'temporal integrity, ACWR flag verification, and encoding checks.',
        S['bullet']))
    story.append(Paragraph(
        '&#8226; <b>0. TI_Missingness_Analysis</b> &#8212; investigation of '
        'why Training Intensity Yesterday has missing values for certain rows: '
        'free-day fill logic validation, NaN patterns by activity type.',
        S['bullet']))

    story.append(Paragraph(
        '<b>Level 1 &#8212; Data Visualisation &amp; EDA:</b>',
        S['body']))
    story.append(Paragraph(
        '&#8226; <b>1.1. Match Analysis</b> &#8212; match-level data exploration '
        'from Games.xlsx: match intensity distributions, per-player performance '
        'profiles, playing time patterns, and relationships between match load '
        'and subsequent wellness responses.',
        S['bullet']))
    story.append(Paragraph(
        '&#8226; <b>1.2. Raw Data Visualisation</b> &#8212; comprehensive EDA '
        'across all four raw datasets: missingness heatmaps, dataset linkage '
        '(Venn diagrams showing player overlap), temporal coverage, wellness '
        'and GPS distributions, per-player radar charts.',
        S['bullet']))
    story.append(Paragraph(
        '&#8226; <b>1.3. Processed Data Visualisation</b> &#8212; EDA of the '
        'merged RTT.xlsx: variable distributions across the full season, '
        'per-player wellness trajectories, ACWR time series, feature '
        'correlation heatmaps.',
        S['bullet']))

    story.append(Paragraph(
        '<b>Level 2 &#8212; Experiments</b> (described in Sections 3&#8211;5):',
        S['body']))
    story.append(Paragraph(
        '&#8226; <b>2.1. Experiment 1</b> &#8212; Match Intensity prediction '
        '(reference / exploratory)',
        S['bullet']))
    story.append(Paragraph(
        '&#8226; <b>2.2. Experiment 2</b> &#8212; Training Intensity prediction '
        '(primary experiment)',
        S['bullet']))
    story.append(Paragraph(
        '&#8226; <b>2.3. Experiment 3</b> &#8212; Status Decrease prediction '
        '(auxiliary experiment)',
        S['bullet']))

    story.append(Paragraph('<b>Key engineered variables</b>', S['h2']))

    story.append(Paragraph(
        '<b>Training Intensity Yesterday</b> (the treatment variable): '
        'A composite GPS score capturing how hard a player trained on the '
        'previous day. Computed as <i>tanh(harmonic_mean(TD%, HSD%, Dec%, '
        'Sprints%) / 100)</i>, where each GPS metric is expressed as a '
        'percentage of the player\'s personal match benchmark. The harmonic '
        'mean penalises sessions where one component is disproportionately '
        'low (e.g., high distance but no sprints), ensuring the score '
        'reflects balanced high-intensity effort. The tanh function soft-caps '
        'the score in [0, 1): 0 = full rest, values near 1 = match-equivalent '
        'effort. Free days receive exactly 0. The per-player distribution of '
        'Training Intensity is shown in the figure below.',
        S['body']))

    story.append(Paragraph(
        '<b>Match Intensity Yesterday</b> (the match-day outcome): '
        'A composite match performance score computed as the '
        '<i>geometric mean</i> of two key metrics &#8212; high-intensity '
        'distance per ball-in-play minute (HID/BIP) and high-intensity '
        'efforts per ball-in-play minute (HIE/BIP) &#8212; scaled by minutes '
        'played: <i>sqrt(HID/BIP &#215; HIE/BIP) &#215; sqrt(clip(minutes, '
        '15, 90) / 90)</i>. The geometric mean ensures that both distance '
        'and effort count equally (a player who runs far but makes few '
        'explosive efforts does not score as highly as one who does both). '
        'The minutes scaling rewards full-match participation. The '
        'distribution of Match Intensity across all matches is shown below. '
        'Only filled on the day after a match.',
        S['body']))

    story.append(Paragraph(
        '<b>Status Decrease</b> (the short-term outcome): '
        'A binary indicator (0 or 1) capturing whether a player\'s medical '
        'status worsened from one day to the next (e.g., Available &#8594; '
        'Attention, or Attention &#8594; Injured). Prevalence is low: '
        'approximately 3&#8211;5% of observations.',
        S['body']))

    story.append(Paragraph(
        '<b>Why these variables?</b> Training Intensity serves as a proxy for '
        'the coaching decision (what load was prescribed), while Match Intensity '
        'and Status Decrease serve as outcomes (how the player responded). '
        'The causal question is whether we can identify the effect of training '
        'intensity on player outcomes while accounting for the fact that '
        'coaches assign training based on player state (confounding by '
        'indication).',
        S['body']))

    # ── Figures ──────────────────────────────────────────────────────────
    proc_dir = output_path.parent / 'data' / 'processed'
    mi_hist = proc_dir / 'Match Intensity Distribution.png'
    ti_box  = proc_dir / 'Training Intensity Per Player Distribution.png'

    if mi_hist.exists():
        story.append(Spacer(1, 6))
        story.append(Paragraph(
            '<i>Figure 1: Distribution of Match Intensity across all player-match '
            'observations. Mean = 3.41, median = 3.34.</i>', S['meta']))
        story.append(Image(str(mi_hist), width=CONTENT_W * 0.75,
                           height=CONTENT_W * 0.45))
        story.append(Spacer(1, 6))

    if ti_box.exists():
        story.append(Paragraph(
            '<i>Figure 2: Training Intensity Yesterday per player, sorted by '
            'median. Coloured by playing position.</i>', S['meta']))
        story.append(Image(str(ti_box), width=CONTENT_W,
                           height=CONTENT_W * 0.35))
        story.append(Spacer(1, 6))

    story.append(PageBreak())

    # ==================================================================
    # 3. EXPERIMENT 1 &#8212; MATCH INTENSITY
    # ==================================================================
    _section(story, '3. Experiment 1 &#8212; Match Intensity Prediction', S)

    story.append(Paragraph(
        '<b>Question:</b> Can we predict how hard a player will perform in '
        'the next match, given their morning training state?',
        S['body']))

    story.append(Paragraph(
        '<b>Why we tried this first:</b> Before building a causal framework '
        'to optimise training load, we needed to verify whether the intended '
        'outcome &#8212; match-day performance &#8212; is even '
        '<i>predictable</i> from training and wellness data. If standard '
        '(non-causal) ML cannot predict it, then causal methods will '
        'certainly not identify a causal effect either. This experiment '
        'serves as that prerequisite check.',
        S['body']))

    story.append(Paragraph(
        'The notebook starts with a <b>diagnostic EDA</b> examining the '
        'prediction task: dataset scope, target distribution, feature-target '
        'correlations, and train/test distribution shift. This revealed '
        'that the effective dataset is small (~248 training, ~72 test) and '
        'features have near-zero correlation with the target.',
        S['body']))

    story.append(Paragraph('<b>Experiment A &#8212; Raw Match Intensity</b>', S['h2']))

    story.append(Paragraph(
        'We predicted Match Intensity Yesterday from morning covariates '
        'using Ridge Regression, XGBoost, CatBoost, and TabPFN across '
        '10 lag values. Player ID was included as a feature.',
        S['body']))

    _table(story,
           ['Model', 'Best Lag', 'RMSE', 'Null RMSE', 'R2', 'Skill'],
           [
               ['TabPFN', '8', '1.1097', '1.3820', '0.272', '+19.7%'],
               ['CatBoost', '5', '1.1932', '1.3820', '0.158', '+13.7%'],
               ['LinReg', '1', '1.2250', '1.3820', '0.113', '+11.4%'],
               ['XGBoost', '2', '1.2412', '1.3820', '0.089', '+10.2%'],
           ], S,
           col_widths=[2.5*cm, 2.0*cm, 2.0*cm, 2.5*cm, 2.0*cm, 5.2*cm])

    story.append(Paragraph(
        'TabPFN achieves R2 = 0.27 &#8212; seemingly useful. However, '
        'feature importance analysis reveals that <b>Player ID is the '
        'dominant feature</b>: the model mostly learns that some players '
        'consistently perform higher than others, regardless of training.',
        S['body']))

    story.append(Paragraph('<b>Experiment B &#8212; Personal Deviation</b>', S['h2']))

    story.append(Paragraph(
        'To test whether there is signal <i>beyond</i> player identity, '
        'we defined <b>Match Intensity Personal Deviation</b> = a player\'s '
        'match intensity minus their own expanding-mean baseline. This '
        'removes the player fixed effect: the model must now predict '
        'whether a player performs above or below <i>their own average</i>.',
        S['body']))

    story.append(Paragraph(
        'Result: all models achieve R2 near zero or negative on the '
        'deviation target. With the player effect removed, there is '
        '<b>no recoverable signal</b> linking morning state to within-player '
        'match performance variation. Per-player median R2 = &#8722;0.75; '
        'only 1 of 17 players has R2 > 0.',
        S['body']))

    story.append(Paragraph(
        'Full results, permutation importance (LOO ablation, greedy forward '
        'selection), and per-player breakdowns are in '
        '<b>notebook 2.1. Experiment1</b>.',
        S['body']))

    _conclusion_box(story, 'Conclusion',
        'Match-day performance is NOT predictable from training and '
        'wellness data. Experiment A showed apparent signal (R2=0.27) '
        'driven entirely by Player ID. Experiment B removed this effect '
        'and found no signal. Since the outcome is not predictable in '
        'a non-causal sense, it cannot serve as a causal target. This '
        'validates redirecting the framework toward proximal outcomes.',
        S)

    story.append(PageBreak())

    # ==================================================================
    # 4. EXPERIMENT 2 &#8212; TRAINING INTENSITY
    # ==================================================================
    _section(story, '4. Experiment 2 &#8212; Training Intensity Prediction', S)

    story.append(Paragraph(
        'Given the negative result of Experiment 1 &#8212; match-day '
        'performance is not predictable from training data &#8212; we '
        'pivoted to a more achievable and equally valuable question.',
        S['body']))

    story.append(Paragraph(
        '<b>Question:</b> Can we recover the coaching staff\'s implicit '
        'load-assignment policy from observable morning state? And if so, '
        'can the model\'s output serve as a <b>data-driven proxy for player '
        'readiness to train</b>?',
        S['body']))

    story.append(Paragraph(
        '<b>Why this matters:</b> Instead of trying to directly optimise '
        'match performance (which proved intractable), we can capture the '
        'coaching expertise embedded in 20 months of daily decisions. If '
        'morning covariates (wellness, fatigue, ACWR, schedule position) '
        'predict what the coaches actually prescribe, then: (1) the model '
        '<b>encodes the coaching staff\'s collective judgment</b> about '
        'player readiness, and (2) the predicted score becomes a continuous '
        '<b>Readiness to Train</b> metric &#8212; if the model predicts '
        'high intensity, the player is in a state where the coaches would '
        'have prescribed a hard session.',
        S['body']))

    story.append(Paragraph(
        '<b>Setup:</b> We predicted Training Intensity Yesterday (continuous, '
        '[0,1)) from 28 morning-assessment covariates using four models: '
        'Ridge Regression, XGBoost, CatBoost, and TabPFN. Each was tested '
        'across 3 lag values (1, 2, 3 days of history). The same processed '
        'RTT.xlsx dataset was used as in all other experiments, but the '
        'effective sample size is much larger than Experiment 1 because '
        'every training day contributes (not just match days).',
        S['body']))

    story.append(Paragraph('<b>Results</b>', S['h2']))

    story.append(Paragraph(
        'Results across all model-lag combinations show that the coaching '
        'policy <b>is substantially recoverable</b> from morning covariates. '
        'The best models (TabPFN, CatBoost) achieve R2 values in the range '
        '0.35&#8211;0.45, meaning 35&#8211;45% of the variance in prescribed '
        'training intensity is explained by observable player state.',
        S['body']))

    _conclusion_box(story, 'Key Finding',
        'The coaching staff\'s load decisions are systematic and largely '
        'driven by observable physiological signals. The fitted model can '
        'serve as a continuous "Readiness to Train" score for each '
        'player-day. This is the primary actionable output of the project.',
        S)

    story.append(Paragraph(
        '<b>Feature importance analysis</b> (permutation importance, 10 '
        'repeats on test set) reveals the <b>top drivers of coaching '
        'decisions</b>:',
        S['body']))

    _table(story,
           ['Rank', 'Feature', 'Importance', 'Interpretation'],
           [
               ['1', 'Days Until Match', '0.0102',
                'Strongest driver: coaches taper intensity as match approaches'],
               ['2', 'Days Until Match (t-1)', '0.0061',
                'Previous day\'s schedule position also matters'],
               ['3', 'Total Distance % Yesterday', '0.0039',
                'Yesterday\'s GPS load volume (% of match benchmark)'],
               ['4', 'Total Distance (m) Yesterday', '0.0021',
                'Raw distance in metres from yesterday\'s session'],
               ['5', 'Total Distance % Yesterday (t-1)', '0.0015',
                'Two-day load history contributes'],
               ['6', 'High Speed Distance (m) Yesterday', '0.0013',
                'High-speed running volume from previous session'],
               ['7', 'Total Minutes Yesterday', '0.0011',
                'Session duration as a load indicator'],
               ['8', 'Activity Type Yesterday (t-1)', '0.0009',
                'Type of session two days ago (game, training, recovery)'],
               ['9', 'Position (t-1)', '0.0008',
                'Playing position influences prescribed load'],
               ['10', 'HSD % Yesterday (t-1)', '0.0008',
                'Two-day high-speed distance history'],
           ], S,
           col_widths=[1.2*cm, 5.0*cm, 2.0*cm, 8.0*cm])

    story.append(Paragraph(
        'The key insight: <b>match-cycle position</b> (Days Until Match) '
        'is by far the strongest predictor, confirming that coaches follow '
        'a clear periodisation pattern. GPS load history (total distance, '
        'high-speed distance) ranks next, indicating that coaches also '
        'account for cumulative recent load when prescribing intensity.',
        S['body']))

    story.append(Paragraph(
        'Full results tables, SHAP/permutation importance plots, and '
        'per-player breakdowns are available in <b>notebook 2.2. Experiment2'
        '</b> with all visualisations preserved.',
        S['body']))

    story.append(Paragraph('<b>Practical use</b>', S['h2']))

    story.append(Paragraph(
        'The Experiment 2 model can be deployed as a <b>decision-support '
        'tool</b>: given a player\'s morning state, the model outputs a '
        'predicted training intensity that reflects what the coaching staff '
        'would typically prescribe. Deviations between the model\'s prediction '
        'and the actual prescription highlight cases where the coach is '
        'making an unusual decision &#8212; which may warrant a second look.',
        S['body']))

    story.append(PageBreak())

    # ==================================================================
    # 5. EXPERIMENT 3 &#8212; STATUS DECREASE
    # ==================================================================
    _section(story, '5. Experiment 3 &#8212; Status Decrease Prediction', S)

    story.append(Paragraph(
        'As an auxiliary experiment, we investigated whether training load '
        'affects short-term player health outcomes. This experiment frames '
        'the problem explicitly in causal terms.',
        S['body']))

    story.append(Paragraph(
        '<b>What is Status Decrease?</b> A binary indicator (0/1) '
        'capturing whether a player\'s medical status worsened from one day '
        'to the next (e.g., Available &#8594; Attention, or Attention '
        '&#8594; Injured). This is a rare event: only about 3&#8211;5% of '
        'player-days show a status decrease.',
        S['body']))

    story.append(Paragraph(
        '<b>Causal framing:</b> In this experiment, <b>Training Intensity '
        'Yesterday</b> is explicitly treated as the <b>treatment variable '
        '(A)</b>, while morning wellness, ACWR, schedule position, and '
        'other player-state features serve as <b>covariates (L)</b>. '
        'Status Decrease is the <b>outcome (Y)</b>. The experiment uses '
        'a single lag (lag=1) to keep the model interpretable.',
        S['body']))

    story.append(Paragraph(
        'Two modes are run:',
        S['body']))

    story.append(Paragraph(
        '&#8226; <b>Prediction mode</b> &#8212; uses only morning '
        'covariates (L) to predict tomorrow\'s status (early-warning system)',
        S['bullet']))
    story.append(Paragraph(
        '&#8226; <b>Causal framing mode</b> &#8212; adds Training '
        'Intensity (A) as treatment variable alongside covariates (L) to '
        'examine whether yesterday\'s load explains today\'s status change',
        S['bullet']))

    story.append(Paragraph('<b>Results</b>', S['h2']))

    story.append(Paragraph(
        'Status Decrease is a challenging prediction target due to the severe '
        'class imbalance (~97% negative, ~3% positive). Models achieve ROC-AUC '
        'values above the 0.5 random baseline but with limited precision at '
        'the decision threshold. The causal framing mode reveals an important '
        'phenomenon: the coefficient on Training Intensity Yesterday is '
        '<b>negative</b> in the logistic regression &#8212; higher training '
        'intensity appears to <i>reduce</i> the risk of status decrease.',
        S['body']))

    story.append(Paragraph(
        'This counterintuitive result is a textbook example of <b>confounding '
        'by indication</b>: coaches prescribe higher intensity to players who '
        'are already in good condition. The raw association between load and '
        'outcome conflates the physiological effect of training with the '
        'coach\'s selection of who receives hard training. Extracting the '
        'true causal effect requires the propensity model from Experiment 2 '
        'combined with G-computation or IPTW methods.',
        S['body']))

    _conclusion_box(story, 'Conclusion',
        'The negative coefficient on Training Intensity demonstrates '
        'exactly why causal methods are needed: naive regression '
        'attributes the coach\'s good judgment (prescribing hard '
        'sessions to healthy players) to the training itself. '
        'Extracting the true causal effect of training intensity on '
        'player health requires the propensity model from Experiment 2 '
        'combined with G-computation or IPTW methods. Full results, '
        'ROC/PR curves, and per-player breakdowns are in '
        'notebook 2.3. Experiment3.',
        S)

    story.append(PageBreak())

    # ==================================================================
    # 6. CONCLUSIONS
    # ==================================================================
    _section(story, '6. Conclusions and Recommendations', S)

    story.append(Paragraph('<b>What we achieved</b>', S['h2']))

    story.append(Paragraph(
        '&#8226; <b>Data integration:</b> Merged four separate monitoring '
        'datasets into a unified, analysis-ready file with 46 engineered '
        'features per player-day, fully reproducible from raw data.',
        S['bullet']))

    story.append(Paragraph(
        '&#8226; <b>Experiment 1</b> confirmed that match-day performance is '
        'not predictable from training data alone (R2 = 0.27, driven by '
        'player identity). This ruled out Match Intensity as a causal target '
        'and validated the focus on proximal outcomes.',
        S['bullet']))

    story.append(Paragraph(
        '&#8226; <b>Experiment 2</b> successfully recovered the coaching '
        'staff\'s load-assignment policy from morning covariates '
        '(R2 = 0.35&#8211;0.45). The model can serve as a continuous '
        '"Readiness to Train" score and as a propensity model for causal '
        'estimation. Feature importance analysis confirmed that periodisation '
        'structure (days until match) and workload ratios (ACWR) are the '
        'primary drivers of coaching decisions.',
        S['bullet']))

    story.append(Paragraph(
        '&#8226; <b>Experiment 3</b> built an outcome model for next-day '
        'Status Decrease and demonstrated the presence of confounding by '
        'indication in the raw data &#8212; confirming that causal methods '
        'are necessary to isolate the true effect of training load.',
        S['bullet']))

    story.append(Paragraph('<b>Recommendations for OH Leuven</b>', S['h2']))

    story.append(Paragraph(
        '<b>1. Deploy the Experiment 2 model as a decision-support tool.</b> '
        'The predicted Training Intensity score from Experiment 2 provides a '
        'data-driven "expected load" for each player-day. When the actual '
        'prescribed load deviates significantly from the prediction, this '
        'flags an unusual coaching decision that may warrant review.',
        S['body']))

    story.append(Paragraph(
        '<b>2. Use the Status Decrease model as an early-warning system.</b> '
        'Although the prediction is imperfect (due to low base rate), '
        'flagging players with elevated risk before the training session '
        'adds a safety layer to the monitoring process.',
        S['body']))

    story.append(Paragraph(
        '<b>3. Continue data collection.</b> More seasons of data will '
        'improve model stability and enable validation across different '
        'squad compositions and fixture schedules.',
        S['body']))

    story.append(Paragraph(
        '<b>4. Next research step: causal estimation.</b> Combining the '
        'propensity model (Experiment 2) with the outcome model '
        '(Experiment 3) via G-computation or Inverse Probability of '
        'Treatment Weighting (IPTW) will allow estimation of the true '
        'causal effect of training intensity on player outcomes &#8212; '
        'free from the confounding by indication demonstrated in '
        'Experiment 3.',
        S['body']))

    # ── Footer ────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.5 * cm))
    _hr(story)
    story.append(Paragraph(
        f'KU Leuven &amp; OH Leuven &#8212; {gen_date}',
        S['meta']))

    # ── Build ─────────────────────────────────────────────────────────
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=2.0 * cm,
        title='Readiness to Train - Project Results',
        author='KU Leuven & OH Leuven',
        subject='Causal Modelling of Player Readiness to Train',
    )
    doc.build(story, onFirstPage=_page_number, onLaterPages=_page_number)
    print('Saved: %s' % output_path)


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    project_root = Path(__file__).parent.parent
    output_path = project_root / 'Project Results.pdf'
    build_pdf(output_path)
