"""
Generate Project Results PDF

Comprehensive results report for the Readiness-to-Train causal ML project.
Summarises findings from all three experiments for delivery to OH Leuven.

Output: Project Results.pdf (project root)

Usage:
    python src/utils/generate_project_results.py
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
    from reportlab.platypus import Image
    S = _styles()
    story = []
    gen_date = datetime.now().strftime('%B %Y')
    project_root = output_path.parent

    # == TITLE ==
    story.append(Spacer(1, 1.8 * cm))
    story.append(Paragraph('Readiness to Train', S['title']))
    story.append(Paragraph('Project Results', S['subtitle']))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph('KU Leuven &amp; OH Leuven', S['meta']))
    story.append(Paragraph(gen_date, S['meta']))
    story.append(Spacer(1, 0.8 * cm))
    _hr(story)

    # ==================================================================
    # 1. INTRODUCTION
    # ==================================================================
    _section(story, '1. Introduction', S)

    story.append(Paragraph(
        'The goal of this project <b>was</b> to develop a <b>causal framework '
        'for optimising training load per player</b> in professional football, '
        'such that each player arrives at match day in the best possible '
        'physical condition. The central question: <i>"Given everything we know '
        'about a player this morning, what training intensity should we prescribe '
        'today to maximise their match-day performance?"</i>',
        S['body']))

    story.append(Paragraph(
        'This is fundamentally a <b>causal</b> question: coaches already make '
        'training decisions based on player state, which means the observed data '
        'confounds the effect of training with the selection of who receives it. '
        'Standard predictive modelling cannot disentangle these; causal methods '
        'are required.',
        S['body']))

    story.append(Paragraph(
        'The project uses daily monitoring data collected by OH Leuven over '
        'approximately 20 months (July 2024 &#8211; February 2026), covering '
        '<b>28 first-team players</b> and <b>5,235 useful player-day observations'
        '</b>. Four raw datasets were merged into a single analysis-ready file '
        '(RTT.xlsx) with 46 engineered features per player-day.',
        S['body']))

    story.append(Paragraph(
        'Two auto-generated data dictionaries document every variable: the '
        '<b>Raw Data Dictionary</b> (<i>data/raw/Raw Data Dictionary.pdf</i>) '
        'covers all columns across the four raw datasets, while the '
        '<b>RTT Data Dictionary</b> (<i>data/processed/RTT Data Dictionary.pdf'
        '</i>) describes all 46 engineered features in the processed dataset, '
        'including their temporal positions and derivations.',
        S['body']))

    story.append(Paragraph(
        'Three experiments were conducted, each building on the previous. '
        'This report tells the story of how we arrived at the results.',
        S['body']))

    # ==================================================================
    # 2. REPOSITORY & NOTEBOOKS
    # ==================================================================
    _section(story, '2. Repository &amp; Notebooks', S)

    story.append(Paragraph(
        'The full project is available on GitHub: '
        '<b><a href="https://github.com/andreasgoethals/Readiness-To-Train" '
        'color="#2E6DA4">github.com/andreasgoethals/Readiness-To-Train</a></b>. '
        'It contains the preprocessing pipeline, five ML model implementations '
        '(Ridge, Logistic Regression, XGBoost, CatBoost, TabPFN), a causal DAG '
        'builder, experiment runner scripts, and all analysis notebooks.',
        S['body']))

    story.append(Paragraph(
        'The notebooks are organised at three levels:',
        S['body']))

    story.append(Paragraph(
        '<b>Level 0 &#8212; Data Quality Checks:</b> '
        '<i>0. Processed_Data_Quality</i> (automated validation of RTT.xlsx) '
        'and <i>0. TI_Missingness_Analysis</i> (Training Intensity NaN patterns).',
        S['bullet']))

    story.append(Paragraph(
        '<b>Level 1 &#8212; Data Visualisation:</b> '
        '<i>1.1. Match Analysis</i> (match-level exploration), '
        '<i>1.2. Raw Data Visualisation</i> (EDA across all 4 raw datasets), '
        '<i>1.3. Processed Data Visualisation</i> (EDA of RTT.xlsx).',
        S['bullet']))

    story.append(Paragraph(
        '<b>Level 2 &#8212; Experiments:</b> '
        '<i>2.1. Experiment 1</i> (Match Intensity), '
        '<i>2.2. Experiment 2</i> (Training Intensity), '
        '<i>2.3. Experiment 3</i> (Status Decrease).',
        S['bullet']))

    story.append(Paragraph(
        'All visualisations are preserved inside the notebooks.',
        S['body']))

    # ==================================================================
    # 3. DATA & KEY VARIABLES
    # ==================================================================
    _section(story, '3. Data &amp; Key Variables', S)

    _table(story,
           ['Dataset', 'Rows', 'Players', 'Days', 'Content'],
           [
               ['Readiness_Data', '14,359', '28', '597',
                'Wellness z-scores, ACWR, GPS benchmarks, medical status'],
               ['Raw_Data', '9,968', '84', '670',
                'Detailed GPS, heart rate, session metadata'],
               ['Sessions', '1,206', '&#8212;', '670',
                'Match day flags, session types'],
               ['Games', '403', '24', '216',
                'High-intensity distance/efforts per ball-in-play, minutes'],
           ], S,
           col_widths=[2.8*cm, 1.4*cm, 1.5*cm, 1.2*cm, 9.3*cm])

    story.append(Paragraph('<b>Training Intensity Yesterday</b> (treatment variable): '
        'A composite GPS score capturing how hard a player trained. Computed as '
        '<i>tanh(harmonic_mean(TD%, HSD%, Dec%, Sprints%) / 100)</i>, where each '
        'GPS metric is a percentage of the player\'s personal match benchmark. '
        'Range [0, 1): 0 = rest, near 1 = match-equivalent effort.',
        S['body']))

    story.append(Paragraph('<b>Match Intensity Yesterday</b> (match-day outcome): '
        'Geometric mean of high-intensity distance and efforts per ball-in-play '
        'minute, scaled by minutes played. Only filled the day after a match.',
        S['body']))

    story.append(Paragraph('<b>Status Decrease</b> (short-term outcome): '
        'Binary indicator of whether a player\'s medical status worsened '
        '(e.g. Available &#8594; Attention). Prevalence ~2%.',
        S['body']))

    # Figures
    proc_dir = project_root / 'data' / 'processed'
    ti_box = proc_dir / 'Training Intensity Per Player Distribution.png'
    mi_hist = proc_dir / 'Match Intensity Distribution.png'

    if ti_box.exists():
        story.append(Spacer(1, 6))
        story.append(Paragraph(
            '<i>Figure 1: Training Intensity per player (sorted by median).</i>',
            S['meta']))
        story.append(Image(str(ti_box), width=CONTENT_W, height=CONTENT_W * 0.35))

    if mi_hist.exists():
        story.append(Spacer(1, 6))
        story.append(Paragraph(
            '<i>Figure 2: Match Intensity distribution (mean = 3.41).</i>',
            S['meta']))
        story.append(Image(str(mi_hist), width=CONTENT_W * 0.75,
                           height=CONTENT_W * 0.45))

    story.append(PageBreak())

    # ==================================================================
    # 4. EXPERIMENT 1
    # ==================================================================
    _section(story, '4. Experiment 1 &#8212; Match Intensity Prediction', S)

    story.append(Paragraph(
        'Before attempting a causal framework, we first needed to verify whether '
        'the intended outcome &#8212; match-day performance &#8212; is even '
        '<i>predictable</i> from training and wellness data. If standard ML '
        'cannot predict it, then causal methods will certainly not identify a '
        'causal effect either: causal identification requires that the outcome '
        'varies systematically with the treatment <i>conditional on confounders'
        '</i>, which is impossible if the outcome has no recoverable association '
        'with the available covariates in the first place.',
        S['body']))

    story.append(Paragraph('<b>Experiment A &#8212; Raw Match Intensity</b>', S['h2']))

    story.append(Paragraph(
        'We predicted Match Intensity Yesterday from 13 morning covariates '
        '(including Player ID) using Ridge Regression, XGBoost, CatBoost, and '
        'TabPFN across 10 lag values. The dataset is small: ~248 training, '
        '~72 test samples (only match-day rows contribute).',
        S['body']))

    _table(story,
           ['Model', 'Best Lag', 'RMSE', 'R2', 'Pearson r'],
           [
               ['LinReg', '1', '1.114', '0.266', '0.723'],
               ['TabPFN', '10', '1.125', '0.251', '0.633'],
               ['CatBoost', '2', '1.194', '0.158', '0.556'],
               ['XGBoost', '1', '1.248', '0.080', '0.613'],
           ], S,
           col_widths=[3.0*cm, 2.5*cm, 2.5*cm, 2.5*cm, 5.7*cm])

    story.append(Paragraph(
        'LinReg achieves R2 = 0.27 at lag=1. However, the <b>permutation '
        'importance analysis</b> reveals that <b>Player ID accounts for almost '
        'all the signal</b> (importance = +0.39, while the next feature is only '
        '+0.01). The model is simply learning that some players consistently '
        'perform at a higher level than others.',
        S['body']))

    story.append(Paragraph(
        '<b>Per-player analysis:</b> All 17 test-set players have negative R2 '
        'except one (R2 = 0.50). The aggregate R2 of 0.27 is entirely driven '
        'by between-player differences, not within-player training effects.',
        S['body']))

    story.append(Paragraph('<b>Experiment B &#8212; Personal Deviation</b>', S['h2']))

    story.append(Paragraph(
        'To confirm that Player ID was the sole driver, we removed the player '
        'fixed effect by predicting <b>Match Intensity Personal Deviation</b> '
        '(each player\'s match intensity minus their own expanding-mean baseline). '
        'This forces the model to predict <i>within-player</i> variation.',
        S['body']))

    story.append(Paragraph(
        'Result: <b>all models achieve R2 near zero or negative</b> across all '
        'lags. The best result (XGBoost lag=5, R2 = &#8722;0.001) is '
        'indistinguishable from random. With the player effect removed, there '
        'is no recoverable signal linking morning state to match performance.',
        S['body']))

    _conclusion_box(story, 'Conclusion',
        'Match-day performance is NOT predictable from training and wellness '
        'data. Experiment A showed apparent signal (R2 = 0.27) driven entirely '
        'by Player ID. Experiment B confirmed no within-player signal exists. '
        'Since the outcome is not predictable in a non-causal sense, it cannot '
        'serve as a causal target. This motivated the pivot to Experiment 2.',
        S)

    story.append(PageBreak())

    # ==================================================================
    # 5. EXPERIMENT 2
    # ==================================================================
    _section(story, '5. Experiment 2 &#8212; Training Intensity Prediction', S)

    story.append(Paragraph(
        'Given that match-day performance is unpredictable, we pivoted to a '
        'different but equally valuable question: <b>can we recover the coaching '
        'staff\'s load-assignment decisions from observable morning state?</b>',
        S['body']))

    story.append(Paragraph(
        'If the answer is yes, the fitted model serves as a <b>data-driven '
        'proxy for player Readiness to Train</b>. The reasoning: if we assume '
        'that the coaches\' decisions were generally <i>good</i> decisions '
        '(informed by years of experience), then a model that accurately '
        'predicts what the coach would prescribe effectively encodes the '
        'coaching staff\'s collective judgment about player readiness. When the '
        'model predicts high intensity, the player is in a state where an '
        'experienced coach would have prescribed a hard session.',
        S['body']))

    story.append(Paragraph(
        '<b>Setup:</b> We predicted Training Intensity Yesterday (continuous, '
        '[0,1)) from 15 morning covariates: Physical State, Mental State, '
        'Total Distance and High Speed Distance ACWR, Any ACWR Danger, '
        'Total Distance % and High Speed Distance % Yesterday, Perceived '
        'Exertion Yesterday, Total Minutes Yesterday, Avg Heart Rate and '
        'Heart Rate Exertion Yesterday, Days Since Game, Days Until Match, '
        'and Player ID. Four models were tested across lags 0&#8211;3. '
        'Dataset: 2,359 training, 634 test observations.',
        S['body']))

    story.append(Paragraph('<b>Results</b>', S['h2']))

    _table(story,
           ['Model', 'Best Lag', 'RMSE', 'R2', 'Pearson r'],
           [
               ['TabPFN', '3', '0.153', '0.430', '0.658'],
               ['CatBoost', '3', '0.158', '0.390', '0.644'],
               ['XGBoost', '2', '0.159', '0.383', '0.629'],
               ['LinReg', '3', '0.194', '0.080', '0.396'],
           ], S,
           col_widths=[3.0*cm, 2.5*cm, 2.5*cm, 2.5*cm, 5.7*cm])

    story.append(Paragraph(
        'The best model (<b>TabPFN, lag=3</b>) achieves <b>R2 = 0.43</b> and '
        '<b>Pearson r = 0.66</b>. This means 43% of the variance in the '
        'coaching staff\'s training intensity decisions is explained by '
        'observable morning-state variables.',
        S['body']))

    story.append(Paragraph(
        '<b>Per-player breakdown:</b> 20 of 22 test-set players have positive '
        'R2 (range 0.14 &#8211; 0.63). Only 2 players show negative R2 '
        '(&#8722;0.27 and &#8722;0.12). The model works consistently across '
        'the squad.',
        S['body']))

    story.append(Paragraph(
        '<b>Feature importance</b> (permutation importance, top 5):',
        S['body']))

    _table(story,
           ['Rank', 'Feature', 'Importance'],
           [
               ['1', 'Days Until Match', '0.0102'],
               ['2', 'Days Until Match (t-1)', '0.0061'],
               ['3', 'Total Distance % Yesterday', '0.0039'],
               ['4', 'Total Distance (m) Yesterday', '0.0021'],
               ['5', 'Total Distance % Yesterday (t-1)', '0.0015'],
           ], S,
           col_widths=[1.5*cm, 7.0*cm, 7.7*cm])

    story.append(Paragraph(
        'Match-cycle position (Days Until Match) is the strongest predictor, '
        'confirming that coaches follow a clear periodisation pattern.',
        S['body']))

    _conclusion_box(story, 'Key Finding',
        'The coaching staff\'s load decisions are systematic and largely driven '
        'by observable physiological signals (R2 = 0.43). The fitted model can '
        'serve as a continuous "Readiness to Train" score for each player-day. '
        'This is the primary actionable output of the project.',
        S)

    story.append(PageBreak())

    # ==================================================================
    # 6. EXPERIMENT 3
    # ==================================================================
    _section(story, '6. Experiment 3 &#8212; Status Decrease Prediction', S)

    story.append(Paragraph(
        'As an auxiliary experiment, we investigated whether we can predict '
        'short-term player health outcomes. <b>Status Decrease</b> is a binary '
        'indicator capturing whether a player\'s medical status worsened from '
        'one day to the next (prevalence ~2%).',
        S['body']))

    story.append(Paragraph(
        '<b>Setup:</b> Logistic Regression, XGBoost, CatBoost, and TabPFN '
        'across 4 lags (0&#8211;3). Covariates include morning wellness, ACWR, '
        'GPS load history, Training Intensity Yesterday, schedule position, and '
        'playing position. Dataset: 3,780 training, 943 test observations '
        '(18 positive cases in test set).',
        S['body']))

    _table(story,
           ['Model', 'Best Lag', 'ROC-AUC', 'Precision', 'Recall', 'F1'],
           [
               ['CatBoost', '2', '0.640', '0.022', '0.444', '0.043'],
               ['LogReg', '2', '0.628', '0.029', '0.611', '0.056'],
               ['XGBoost', '2', '0.616', '0.069', '0.111', '0.085'],
               ['TabPFN', '3', '0.599', '0.000', '0.000', '0.000'],
           ], S,
           col_widths=[2.5*cm, 2.0*cm, 2.2*cm, 2.2*cm, 2.2*cm, 5.1*cm])

    story.append(Paragraph(
        'Status Decrease is extremely challenging to predict due to the severe '
        'class imbalance (~98% negative, ~2% positive, only 18 positive cases '
        'in the test set). The best model (CatBoost, AUC = 0.64) shows some '
        'discriminative ability above random (0.5) but with very low precision.',
        S['body']))

    _conclusion_box(story, 'Conclusion',
        'Status Decrease prediction shows modest signal (AUC = 0.64) but is '
        'limited by extreme class imbalance. The model could serve as a '
        'supplementary early-warning flag but should not be relied upon alone.',
        S)

    story.append(PageBreak())

    # ==================================================================
    # 7. CONCLUSIONS
    # ==================================================================
    _section(story, '7. Conclusions &amp; Recommendations', S)

    story.append(Paragraph('<b>What we achieved</b>', S['h2']))

    story.append(Paragraph(
        '&#8226; <b>Data integration:</b> Merged four monitoring datasets into '
        'a unified, analysis-ready file with 46 features per player-day.',
        S['bullet']))

    story.append(Paragraph(
        '&#8226; <b>Experiment 1</b> confirmed that match-day performance is '
        'not predictable from training data (R2 driven by Player ID; personal '
        'deviation R2 near zero). This ruled out Match Intensity as a causal '
        'target.',
        S['bullet']))

    story.append(Paragraph(
        '&#8226; <b>Experiment 2</b> successfully recovered the coaching '
        'staff\'s load-assignment policy (best R2 = 0.43, Pearson r = 0.66). '
        'The model serves as a data-driven Readiness to Train proxy, encoding '
        'the coaching staff\'s collective expertise.',
        S['bullet']))

    story.append(Paragraph(
        '&#8226; <b>Experiment 3</b> demonstrated modest ability to predict '
        'Status Decrease (AUC = 0.64), limited by extreme class imbalance.',
        S['bullet']))

    story.append(Paragraph('<b>Recommendations for OH Leuven</b>', S['h2']))

    story.append(Paragraph(
        '<b>1.</b> Deploy the Experiment 2 model as a <b>decision-support tool'
        '</b>. The predicted Training Intensity provides a data-driven '
        '"expected load" for each player-day. Deviations between model '
        'prediction and actual prescription flag unusual decisions.',
        S['body']))

    story.append(Paragraph(
        '<b>2.</b> Continue data collection. More seasons will improve model '
        'stability and enable validation across different squad compositions '
        'and fixture schedules.',
        S['body']))

    # Footer
    story.append(Spacer(1, 0.5 * cm))
    _hr(story)
    story.append(Paragraph(f'KU Leuven &amp; OH Leuven &#8212; {gen_date}', S['meta']))

    # Build
    doc = SimpleDocTemplate(
        str(output_path), pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=2.0 * cm,
        title='Readiness to Train - Project Results',
        author='KU Leuven & OH Leuven',
    )
    doc.build(story, onFirstPage=_page_number, onLaterPages=_page_number)
    print('Saved: %s' % output_path)


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    project_root = Path(__file__).parent.parent.parent
    output_path = project_root / 'Project Results.pdf'
    build_pdf(output_path)
