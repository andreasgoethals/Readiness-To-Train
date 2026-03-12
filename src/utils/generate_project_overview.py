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
    S = _styles()
    story = []

    # ── Cover ─────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 1.8 * cm))   # push title down from top margin
    story.append(Paragraph('Readiness to Train', S['title']))
    # spaceAfter=16 on title + spaceBefore=6 on subtitle prevents overlap
    story.append(Paragraph(
        'Prescriptive Analytics for Optimal Training Intensity', S['subtitle']))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph(
        'KU Leuven &amp; OH Leuven &#8212; PhD Research Project', S['meta']))
    story.append(Paragraph('Project Overview &amp; Problem Statement', S['meta']))
    story.append(Spacer(1, 0.8 * cm))
    _hr(story)
    story.append(Spacer(1, 0.4 * cm))

    # ── 1. Core Objective ─────────────────────────────────────────────────────
    _section(story, '1. Core Objective', S)

    story.append(Paragraph(
        'This project develops a <b>prescriptive analytics system</b> that estimates '
        'individual player <b>Readiness to Train</b> in professional football. Using '
        '<b>Causal Machine Learning</b> within a <b>Dynamic Treatment Regime (DTR)</b> '
        'framework, the system recommends an optimal daily training intensity for each '
        'player &#8212; delivered as a continuous score between 0 and 1. A score of 0 '
        'indicates the player should rest entirely; a score of 1 indicates the player '
        'can train at maximal intensity.',
        S['body']))

    story.append(Paragraph(
        'The system is fully <b>individualised</b>: each player receives a '
        'personalised score derived from their own physiological profile, training '
        'history, and evolving daily state. The problem is inherently a '
        '<b>Dynamic Treatment Regime with variable-length decision horizons</b> '
        'because (1) not every player is selected for every match &#8212; so some '
        'players skip a cycle entirely while others play &#8212; and (2) the time '
        'between consecutive matches varies (typically 5&#8211;8 days, but '
        'sometimes more during international breaks or cup fixtures). Each player\'s '
        'match-cycle length therefore differs both across players and across time.',
        S['body']))

    _callout(story,
             'The objective is NOT injury minimisation. A policy that minimises '
             'injury risk trivially prescribes zero load. The true objective is '
             'performance optimisation under biological constraints &#8212; where some '
             'non-zero injury incidence may be optimal because competitive '
             'performance requires physiological stress.',
             S)

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
           ['Dataset', 'Rows', 'Columns', 'Players', 'Date Range', 'Granularity'],
           [
               ['Readiness_Data.xlsx', '14,359', '24', '28',
                '2024-07-01 \u2192 2026-02-17\n(597 days)', 'Daily (player-day)'],
               ['Raw_Data.xlsx', '9,968', '39', '84',
                '2024-05-02 \u2192 2026-03-01', 'Session-level (player-session)'],
               ['Sessions.xlsx', '1,206', '9', '\u2014',
                '2024-05-02 \u2192 2026-03-01', 'Session-level (team-session)'],
               ['Games.xlsx', '403', '9', '24',
                '2025-07-27 \u2192 2026-02-28\n(27 match dates)', 'Match-level (player-match)'],
           ], S,
           col_widths=[3.2*cm, 1.3*cm, 1.9*cm, 1.8*cm, 3.8*cm, 4.2*cm])

    story.append(Paragraph(
        'The <b>processed dataset</b> merges all four sources. Readiness_Data is the '
        'base; Raw_Data provides detailed GPS/HR at session level (shifted +1 day so '
        'it appears as "yesterday" data); Games provides match performance data '
        '(also shifted +1 day). Player overlap: all 28 Readiness_Data players appear '
        'in Raw_Data; 23 of 28 appear in Games.',
        S['body']))

    # ── 3. Why This Is a Causal Problem ──────────────────────────────────────
    _section(story, '3. Why This Is a Causal Problem', S)

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

    # ── 4. Causal Framework Variables ─────────────────────────────────────────
    _section(story, '4. Causal Framework Variables', S)

    _table(story,
           ['Symbol', 'Role', 'Variables'],
           [
               ['L(t)  (Covariates)',
                'Player state at time t',
                'Wellness z-scores (fatigue, readiness, soreness, sleep, stress, '
                'mood), composite scores (Physical State, Mental State, Overall '
                'Wellbeing), ACWR metrics, Days Since Game, Days Until Match, '
                'medical availability, club attendance, raw GPS/HR (yesterday), '
                'match performance (yesterday &#8212; only filled day after match)'],
               ['A(t)  (Treatment)',
                'Daily training intensity',
                'Continuous score in [0,1], derived as tanh(mean of Total Distance %, '
                'High-Speed Distance %, Decelerations %, Sprints %) normalised '
                'against individual match benchmarks. GPS % columns are capped at '
                '250% to remove extreme outliers. Free/rest days \u2192 0.'],
               ['Y  (Outcome)',
                'Match-day performance',
                'High-intensity distance per Ball-In-Play minute (m/BIP). '
                'Continuous, to be maximised. Only filled in Games dataset on '
                'match-day rows (shifted +1 day in processed data).'],
           ], S,
           col_widths=[2.8*cm, 3.2*cm, 10.2*cm])

    # ── 5. Temporal Semantics ─────────────────────────────────────────────────
    _section(story, '5. Temporal Semantics (Critical)', S)

    story.append(Paragraph(
        'Each row in the dataset represents <b>one player-day</b> (date t). '
        'Variables within a single row have different temporal positions. '
        'Understanding this is essential for avoiding data leakage and correctly '
        'specifying causal models.',
        S['body']))

    _table(story,
           ['Temporal position', 'Variables', 'Available at prediction?'],
           [
               ['Before day t\n(t\u22121 data)',
                'Activity Type Yesterday, GPS % (\u00d75), Training Intensity '
                'Yesterday, ACWR (\u00d74), RPE Yesterday, Comment Yesterday, '
                'Total Minutes / Distance / HS Distance / HR / HR Exertion '
                'Yesterday (from Raw_Data), Match High Intensity / HIT Efforts / '
                'Minutes Played Yesterday (from Games, day after match only)',
                'Yes \u2014 fully known'],
               ['Morning of day t\n(assessment)',
                'Wellness z-scores (fatigue, readiness, soreness, sleep quality, '
                'stress, mood), Physical State, Mental State, Overall Wellbeing, '
                'Status, Status Decrease, Days Since Game, Days Until Match, '
                'Match Day, Medical Availability, Club Attendance, Position',
                'Yes \u2014 measured before activity decisions'],
               ['Post-assessment\n(same day t)',
                'Activity Type Today, Selected',
                'No \u2014 assigned AFTER morning assessment. Only valid as treatment '
                'A(t) in causal analyses, never as a predictor in standard ML.'],
           ], S,
           col_widths=[3.2*cm, 8.8*cm, 4.2*cm])

    story.append(Paragraph('<b>Critical rules:</b>', S['body']))

    story.append(Paragraph(
        '<b>Days Since Game is NEVER 0.</b> It counts days since the last '
        'completed OHL first-team game. On match day the game has not yet '
        'occurred at the morning assessment, so the count refers to the '
        'previous match. Minimum value: 1.',
        S['bullet']))

    story.append(Paragraph(
        '<b>Days Until Match CAN be 0</b> &#8212; on match day itself for '
        'selected players. Only OHL first-team matches are counted.',
        S['bullet']))

    story.append(Paragraph(
        '<b>Match Day is schedule information</b> &#8212; although derived from '
        'Activity Type Today in preprocessing, it represents the team fixture '
        'list (known in advance) and is safe to use as a predictor.',
        S['bullet']))

    story.append(Paragraph(
        '<b>Wellness z-scores are individualised</b> &#8212; each player\'s '
        'z-scores are relative to their own 28-day rolling baseline, not the '
        'team average.',
        S['bullet']))

    story.append(Paragraph(
        '<b>Lag features are player-grouped</b> &#8212; shifted values '
        '(e.g. Fatigue (z) at t&#8722;1) are created within each player\'s own '
        'time series to prevent cross-player data leakage.',
        S['bullet']))

    # ── 6. Sports Science Foundation ──────────────────────────────────────────
    _section(story, '6. Sports Science Foundation', S)

    story.append(Paragraph('<b>Supercompensation</b>', S['h2']))
    story.append(Paragraph(
        'Training stress produces fatigue, which, during recovery, is '
        'followed by adaptation and an increase in performance capacity above '
        'baseline. This cycle &#8212; stress &#8594; fatigue &#8594; recovery '
        '&#8594; supercompensation &#8212; underpins all periodisation theory. '
        'The DTR framework seeks the sequence of training intensities that '
        'maximises supercompensation at match day, neither under-loading '
        '(insufficient stimulus) nor over-loading (accumulated fatigue with '
        'insufficient recovery).',
        S['body']))

    story.append(Paragraph('<b>Acute:Chronic Workload Ratio (ACWR)</b>', S['h2']))
    story.append(Paragraph(
        'ACWR is the ratio of a player\'s recent acute load (7-day exponential '
        'moving average) to their chronic load (42-day EMA). It captures where '
        'a player sits on the stress-recovery-adaptation curve. The dataset '
        'provides ACWR for four GPS metrics: Total Distance, High-Speed '
        'Distance, Decelerations (&gt;3 m/s&#178;), and Sprints.',
        S['body']))

    _table(story,
           ['ACWR Zone', 'Interpretation'],
           [
               ['&lt; 0.8', 'Under-trained &#8212; insufficient stimulus for adaptation'],
               ['0.8 &#8211; 1.3', 'Sweet spot &#8212; optimal fitness-fatigue balance'],
               ['1.3 &#8211; 1.5', 'Caution &#8212; elevated fatigue accumulation'],
               ['&gt; 1.5', 'Danger zone &#8212; flagged as Any ACWR Danger = 1; '
                'injury risk sharply elevated (Gabbett, 2016)'],
           ], S,
           col_widths=[3.0*cm, 13.2*cm])

    story.append(Paragraph('<b>Individual Profiling</b>', S['h2']))
    story.append(Paragraph(
        'GPS metrics are expressed as a percentage of each player\'s own '
        'match-day benchmarks (e.g. Total Distance % = session distance / '
        'player\'s average match distance). This normalises for positional '
        'differences and individual fitness levels, making comparisons across '
        'players meaningful. Values are capped at 250% to remove extreme '
        'outliers from low-benchmark or early-season matches. Wellness z-scores '
        'use a 28-day rolling window per player. The training intensity score '
        'A(t) is the tanh-compressed mean of these GPS percentages, '
        'soft-capped in [0,&nbsp;1).',
        S['body']))

    story.append(Paragraph('<b>Match Cycle Structure</b>', S['h2']))
    story.append(Paragraph(
        'The typical weekly structure consists of 5&#8211;6 training days '
        'followed by a match day. This creates the <b>match cycle</b> structure '
        'central to the DTR formulation. Each cycle begins the day after a match '
        'and ends on the next match day. Cycle boundaries are '
        '<b>player-specific</b>: days where the team plays but a given player '
        'is not selected simply extend that player\'s current cycle.',
        S['body']))

    # ── 7. Key Challenges ─────────────────────────────────────────────────────
    _section(story, '7. Key Challenges', S)

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

    # ── 8. Research Questions ─────────────────────────────────────────────────
    _section(story, '8. Key Research Questions', S)

    story.append(Paragraph(
        '<b>RQ1.</b> How can we accurately estimate the effect of training '
        'intensity sequences in a "Small N, Large T" observational environment '
        '(28 players, up to 597 days) where standard statistical assumptions '
        'are violated by time-varying confounding?',
        S['bullet']))

    story.append(Paragraph(
        '<b>RQ2.</b> How do we handle <b>time-varying confounding affected by '
        'prior treatment</b> in observational football data, where coaching '
        'decisions create systematic confounding between player state and '
        'prescribed load?',
        S['bullet']))

    story.append(Paragraph(
        '<b>RQ3.</b> Can a causal DTR model identify the optimal '
        'individualised training intensity sequence &#8212; one per day in the '
        'pre-match cycle &#8212; that maximises expected match-day performance '
        'for each player?',
        S['bullet']))

    # ── 9. What Is NOT This Project ───────────────────────────────────────────
    _section(story, '9. What This Project Is Not', S)

    _table(story,
           ['This project IS', 'This project is NOT'],
           [
               ['Prescriptive analytics &#8212; recommending optimal load',
                'Predictive injury modelling &#8212; predicting injury occurrence'],
               ['Causal inference &#8212; estimating treatment effects',
                'Association mining &#8212; correlating load with outcomes'],
               ['Individualised &#8212; one DTR per player',
                'Team-level &#8212; aggregate group recommendations'],
               ['Sequential &#8212; optimising a full pre-match cycle',
                'Single-stage &#8212; one-shot intervention recommendation'],
               ['Performance optimisation &#8212; maximising match output',
                'Risk minimisation &#8212; zero-load policies are trivially "safe"'],
               ['G-methods / DTR methods for time-varying confounding',
                'Causal meta-learners (S/T/X/DR-Learner) &#8212; single-stage only'],
           ], S,
           col_widths=[8.0*cm, 8.2*cm])

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
