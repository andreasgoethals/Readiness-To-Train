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
    HRFlowable, KeepTogether, PageBreak
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
    S = _styles()
    story = []

    # ── Title page ───────────────────────────────────────────────────────
    story.append(Spacer(1, 1.8 * cm))
    story.append(Paragraph('Readiness to Train', S['title']))
    story.append(Paragraph('Project Results', S['subtitle']))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph('KU Leuven &amp; OH Leuven', S['meta']))
    story.append(Paragraph('March 2026', S['meta']))
    story.append(Spacer(1, 0.8 * cm))
    _hr(story)
    story.append(Spacer(1, 0.4 * cm))

    # ==================================================================
    # 1. INTRODUCTION
    # ==================================================================
    _section(story, '1. Introduction', S)

    story.append(Paragraph(
        'This report summarises the results of the <b>Readiness to Train</b> '
        'project: a collaboration between KU Leuven and OH Leuven to build '
        'a data-driven system that helps coaching staff optimise daily '
        'training intensity for each player. The goal is to answer the '
        'question: <i>"Given everything we know about a player this morning, '
        'what training intensity should we prescribe today?"</i>',
        S['body']))

    story.append(Paragraph(
        'The project uses <b>daily monitoring data</b> collected by OH Leuven '
        'over approximately 20 months (July 2024 &#8211; February 2026), '
        'covering <b>28 first-team players</b> and totalling '
        '<b>14,359 player-day observations</b>. Four raw datasets were '
        'merged into a single analysis-ready file containing 46 engineered '
        'features per player-day.',
        S['body']))

    story.append(Paragraph(
        'Three experiments were conducted, each answering a different '
        'question. Before describing the experiments, we explain the key '
        'variables that were constructed during preprocessing.',
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
        'Before any modelling, we conducted thorough exploratory data analysis '
        'across several notebooks:',
        S['body']))

    story.append(Paragraph(
        '&#8226; <b>Raw Data Visualisation</b> &#8212; examined all four '
        'datasets individually: variable distributions, missingness patterns, '
        'player overlap across datasets (Venn diagrams), temporal coverage, '
        'per-player radar charts for wellness and GPS profiles, and '
        'cross-dataset correlations.',
        S['bullet']))

    story.append(Paragraph(
        '&#8226; <b>Processed Data Visualisation</b> &#8212; explored the '
        'merged RTT.xlsx dataset: feature distributions across the full season, '
        'per-player wellness trajectories, ACWR time series, and feature '
        'correlation heatmaps to identify redundancy and multicollinearity.',
        S['bullet']))

    story.append(Paragraph(
        '&#8226; <b>Match Analysis</b> &#8212; dedicated exploration of match-level '
        'data from Games.xlsx: match intensity distributions, per-player performance '
        'profiles, playing time patterns, and the relationship between match load and '
        'subsequent wellness responses.',
        S['bullet']))

    story.append(Paragraph(
        '&#8226; <b>Data quality checks</b> &#8212; automated validation of the '
        'processed dataset (column completeness, temporal integrity, ACWR flags) '
        'and targeted investigation of Training Intensity missingness patterns.',
        S['bullet']))

    story.append(Paragraph('<b>Key engineered variables</b>', S['h2']))

    story.append(Paragraph(
        '<b>Training Intensity Yesterday</b> (the treatment variable): '
        'A composite GPS score capturing how hard a player trained on the '
        'previous day. Computed as <i>tanh(mean(TD%, HSD%, Dec%, Sprints%) '
        '/ 100)</i>, where each GPS metric is expressed as a percentage of '
        'the player\'s personal match benchmark. The tanh function soft-caps '
        'the score in [0, 1). A value of 0 means full rest; values near 1 '
        'approach match-equivalent effort. Free days (no training) receive a '
        'score of exactly 0.',
        S['body']))

    story.append(Paragraph(
        '<b>Match Intensity Yesterday</b> (the match-day outcome): '
        'A composite match performance score computed as '
        '<i>sqrt(HID/BIP &#215; HIE/BIP) &#215; sqrt(clip(minutes, 15, 90) '
        '/ 90)</i>, where HID/BIP is high-intensity distance per ball-in-play '
        'minute and HIE/BIP is high-intensity efforts per ball-in-play minute. '
        'This captures both the intensity and duration of a player\'s match '
        'contribution. Only filled on the day after a match.',
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
        '<b>Why we tried this:</b> If match-day performance were predictable '
        'from training data, we could directly optimise the training regime '
        'to maximise match output. This would be the ideal causal target '
        'for the Dynamic Treatment Regime framework.',
        S['body']))

    story.append(Paragraph(
        '<b>Setup:</b> We predicted Match Intensity Yesterday (continuous) '
        'from morning covariates using four models: Ridge Regression, '
        'XGBoost, CatBoost, and TabPFN. Each model was tested across 10 lag '
        'values (1&#8211;10 days of history). The dataset is small &#8212; only '
        'rows following a match contribute (~248 training, ~72 test samples).',
        S['body']))

    story.append(Paragraph('<b>Results</b>', S['h2']))

    _table(story,
           ['Model', 'Best Lag', 'RMSE', 'Null RMSE', 'R2', 'Skill vs Null'],
           [
               ['TabPFN', '8', '1.1097', '1.3820', '0.272', '+19.7%'],
               ['CatBoost', '5', '1.1932', '1.3820', '0.158', '+13.7%'],
               ['LinReg', '1', '1.2250', '1.3820', '0.113', '+11.4%'],
               ['XGBoost', '2', '1.2412', '1.3820', '0.089', '+10.2%'],
           ], S,
           col_widths=[2.5*cm, 2.0*cm, 2.0*cm, 2.5*cm, 2.0*cm, 5.2*cm])

    story.append(Paragraph(
        'The best model (TabPFN) achieves R2 = 0.27, meaning it explains '
        'about 27% of the variance in match intensity. While this is better '
        'than the null baseline (predicting the mean for everyone), the '
        '<b>per-player analysis reveals the problem</b>:',
        S['body']))

    story.append(Paragraph(
        '&#8226; Median per-player R2 = <b>-0.75</b> (the model is '
        '<i>worse</i> than predicting the mean for most players)',
        S['bullet']))
    story.append(Paragraph(
        '&#8226; Only <b>1 out of 17</b> test-set players has R2 > 0',
        S['bullet']))
    story.append(Paragraph(
        '&#8226; The aggregate R2 of 0.27 is driven by '
        'Player ID being the most important feature &#8212; some players '
        'consistently perform higher than others regardless of training',
        S['bullet']))

    story.append(Paragraph(
        'Adding more historical lags barely helps: the maximum improvement '
        'from lag=1 to the best lag is only 1.6%. This confirms that the '
        'morning-of-match-day state captures virtually all recoverable signal.',
        S['body']))

    story.append(Paragraph('<b>Conclusion</b>', S['h2']))

    _callout(story,
             'Match-day performance is NOT predictable from training and '
             'wellness data alone. Too many unobserved factors intervene: '
             'tactical decisions, opponent quality, team composition, player '
             'psychology. Since the outcome is not even predictable in a '
             'standard (non-causal) ML sense, it will certainly not be '
             'identifiable as a causal target either. This validates the '
             'decision to focus on more proximal outcomes.',
             S)

    story.append(PageBreak())

    # ==================================================================
    # 4. EXPERIMENT 2 &#8212; TRAINING INTENSITY
    # ==================================================================
    _section(story, '4. Experiment 2 &#8212; Training Intensity Prediction', S)

    story.append(Paragraph(
        '<b>Question:</b> Can we recover the coaching staff\'s implicit '
        'load-assignment policy from observable morning state?',
        S['body']))

    story.append(Paragraph(
        '<b>Why this matters:</b> If morning covariates (wellness, fatigue, '
        'ACWR, days since/until match) predict the training intensity that '
        'coaches actually prescribe, two things follow: (1) the coaching '
        'decision-making is <b>systematic and data-driven</b>, and (2) the '
        'fitted model can serve as a proxy for <b>player readiness to train'
        '</b> &#8212; if the model predicts high intensity, the player is in '
        'a state where the coaches would have prescribed a hard session.',
        S['body']))

    story.append(Paragraph(
        '<b>Setup:</b> We predicted Training Intensity Yesterday (continuous, '
        '[0,1)) from 28 morning-assessment covariates using four models: '
        'Ridge Regression, XGBoost, CatBoost, and TabPFN. Each was tested '
        'across 3 lag values (1, 2, 3 days of history). The full dataset was '
        'used (~14,000 rows after lag creation), giving much larger train and '
        'test sets than Experiment 1.',
        S['body']))

    story.append(Paragraph('<b>Results</b>', S['h2']))

    story.append(Paragraph(
        'Results across all model-lag combinations show that the coaching '
        'policy <b>is substantially recoverable</b> from morning covariates. '
        'The best models (TabPFN, CatBoost) achieve R2 values in the range '
        '0.35&#8211;0.45, meaning 35&#8211;45% of the variance in prescribed '
        'training intensity is explained by observable player state.',
        S['body']))

    _finding(story,
             'Key finding: the coaching staff\'s load decisions are systematic '
             'and largely driven by observable physiological signals. The '
             'fitted model can serve as a continuous "Readiness to Train" '
             'score for each player-day.',
             S)

    story.append(Paragraph(
        '<b>Feature importance analysis</b> (using permutation importance and '
        'SHAP) reveals the <b>top drivers of coaching decisions</b>:',
        S['body']))

    story.append(Paragraph(
        '&#8226; <b>Days Until Match / Days Since Game</b> &#8212; match-cycle '
        'position is the single strongest predictor. Coaches follow a clear '
        'periodisation pattern: intensity peaks mid-cycle and tapers as the '
        'match approaches.',
        S['bullet']))

    story.append(Paragraph(
        '&#8226; <b>ACWR (Acute:Chronic Workload Ratio)</b> &#8212; coaches '
        'reduce intensity when ACWR is elevated, consistent with Gabbett\'s '
        'training-injury prevention framework.',
        S['bullet']))

    story.append(Paragraph(
        '&#8226; <b>Previous-day training intensity</b> &#8212; strong '
        'autocorrelation indicates coaches plan multi-day loading sequences '
        'rather than making independent daily decisions.',
        S['bullet']))

    story.append(Paragraph(
        '&#8226; <b>Wellness (fatigue, readiness, soreness)</b> &#8212; '
        'self-reported physical state modulates load prescription, though '
        'its importance is secondary to schedule position.',
        S['bullet']))

    story.append(Paragraph('<b>Practical use</b>', S['h2']))

    story.append(Paragraph(
        'The Experiment 2 model can be deployed as a <b>decision-support '
        'tool</b>: given a player\'s morning state, the model outputs a '
        'predicted training intensity that reflects what the coaching staff '
        'would typically prescribe. Deviations between the model\'s prediction '
        'and the actual prescription highlight cases where the coach is '
        'making an unusual decision &#8212; which may warrant a second look. '
        'Additionally, the model serves as a <b>propensity model</b> for '
        'downstream causal estimation (IPTW weighting).',
        S['body']))

    story.append(PageBreak())

    # ==================================================================
    # 5. EXPERIMENT 3 &#8212; STATUS DECREASE
    # ==================================================================
    _section(story, '5. Experiment 3 &#8212; Status Decrease Prediction', S)

    story.append(Paragraph(
        '<b>Question:</b> Can we predict which players are at risk of a '
        'status deterioration tomorrow?',
        S['body']))

    story.append(Paragraph(
        '<b>What is Status Decrease?</b> A binary indicator (yes/no) '
        'capturing whether a player\'s medical status worsened from one day '
        'to the next. For example, going from "Available" to "Attention" or '
        'from "Attention" to "Injured" counts as a status decrease. This is '
        'a rare event &#8212; only about 3&#8211;5% of player-days show a '
        'status decrease.',
        S['body']))

    story.append(Paragraph(
        '<b>Setup:</b> We used Logistic Regression, XGBoost, and CatBoost '
        'to predict Status Decrease from morning covariates, running in two '
        'modes:',
        S['body']))

    story.append(Paragraph(
        '&#8226; <b>Prediction mode</b> &#8212; uses only morning state to '
        'predict tomorrow\'s status (early-warning system)',
        S['bullet']))
    story.append(Paragraph(
        '&#8226; <b>Causal framing mode</b> &#8212; adds Training Intensity '
        'as a covariate to examine whether yesterday\'s load explains today\'s '
        'status change (diagnostic analysis)',
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

    _callout(story,
             'The negative coefficient on Training Intensity demonstrates '
             'exactly why causal methods are needed: naive regression '
             'attributes the coach\'s good judgment (prescribing hard '
             'sessions to healthy players) to the training itself.',
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
        'KU Leuven &amp; OH Leuven &#8212; March 2026',
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
    story.append(Spacer(1, 0.6 * cm))


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    project_root = Path(__file__).parent.parent
    output_path = project_root / 'Project Results.pdf'
    build_pdf(output_path)
