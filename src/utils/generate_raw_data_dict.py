"""
Standalone script to generate the Raw Data Dictionary PDF.
Saved to: data/raw/Raw Data Dictionary.pdf

Run from the project root:
    python scripts/generate_raw_data_dict.py
"""
from pathlib import Path
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Table, TableStyle,
    Spacer, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
OUT  = ROOT / "data" / "raw" / "Raw Data Dictionary.pdf"


# ---------------------------------------------------------------------------
# Colour palette (one colour per dataset)
# ---------------------------------------------------------------------------
C_NAVY  = colors.HexColor("#1a1a2e")
C_RD    = colors.HexColor("#1565C0")   # Readiness_Data
C_RAW   = colors.HexColor("#00695C")   # Raw_Data
C_SESS  = colors.HexColor("#6A1B9A")   # Sessions
C_GAMES = colors.HexColor("#AD1457")   # Games
C_BG    = colors.HexColor("#fafafa")


# ---------------------------------------------------------------------------
# Paragraph styles
# ---------------------------------------------------------------------------
styles    = getSampleStyleSheet()
title_s   = ParagraphStyle("T",   parent=styles["Heading1"],  fontSize=18,
                textColor=C_NAVY, spaceAfter=6, fontName="Helvetica-Bold",
                alignment=TA_CENTER)
sub_s     = ParagraphStyle("S",   parent=styles["Normal"],    fontSize=9,
                leading=12, textColor=colors.HexColor("#555555"),
                spaceAfter=14, alignment=TA_CENTER)
h2_s      = ParagraphStyle("H2",  parent=styles["Heading2"],  fontSize=12,
                textColor=C_NAVY, spaceBefore=14, spaceAfter=6,
                fontName="Helvetica-Bold")
note_s    = ParagraphStyle("N",   parent=styles["Normal"],    fontSize=8.5,
                leading=12, textColor=colors.HexColor("#333333"), spaceAfter=6)
small_s   = ParagraphStyle("Sm",  parent=styles["Normal"],    fontSize=8,
                leading=11, textColor=colors.HexColor("#444444"),
                spaceBefore=4, spaceAfter=16)
cell_s    = ParagraphStyle("C",   parent=styles["Normal"],    fontSize=8,  leading=10)
hdr_s     = ParagraphStyle("H",   parent=styles["Normal"],    fontSize=8.5,
                fontName="Helvetica-Bold", textColor=colors.white, leading=11)
def_s     = ParagraphStyle("D",   parent=styles["Normal"],    fontSize=8.5,
                leading=13, textColor=colors.HexColor("#333333"),
                spaceBefore=3, spaceAfter=3)
tiny_s    = ParagraphStyle("Tiny",parent=styles["Normal"],    fontSize=8,
                leading=11, textColor=colors.HexColor("#666666"),
                spaceBefore=4, spaceAfter=14)


# ---------------------------------------------------------------------------
# Helper: build a column-schema table
# ---------------------------------------------------------------------------
def ds_table(cols, color):
    """
    cols: list of (name, type, description) tuples
    color: header / name-column background colour
    """
    hdr = [
        Paragraph("<b>Column</b>",      hdr_s),
        Paragraph("<b>Type</b>",        hdr_s),
        Paragraph("<b>Description</b>", hdr_s),
    ]
    data = [hdr] + [
        [Paragraph(r[0], cell_s), Paragraph(r[1], cell_s), Paragraph(r[2], cell_s)]
        for r in cols
    ]
    t = Table(data, colWidths=[2.0 * inch, 0.85 * inch, 3.75 * inch])
    cmds = [
        ("BACKGROUND",    (0, 0), (-1, 0), color),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",   (0, 0), (-1, -1), 5),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 5),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("BACKGROUND",    (0, 1), (-1, -1), C_BG),
    ]
    # Alternate row striping
    for i in range(1, len(data), 2):
        cmds.append(("BACKGROUND", (0, i), (-1, i), colors.white))
    t.setStyle(TableStyle(cmds))
    return t


# ---------------------------------------------------------------------------
# Build document
# ---------------------------------------------------------------------------
def build_pdf():
    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4,
        leftMargin=0.6 * inch, rightMargin=0.6 * inch,
        topMargin=0.6 * inch,  bottomMargin=0.6 * inch,
    )
    el = []

    # ── Title ────────────────────────────────────────────────────────────────
    el.append(Paragraph("Raw Data Dictionary", title_s))
    el.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d')}"
        " &nbsp;|&nbsp; Project: Readiness to Train"
        " &nbsp;|&nbsp; Partnership: KU Leuven &amp; OH Leuven",
        sub_s,
    ))

    # ── Overview table ────────────────────────────────────────────────────────
    el.append(Paragraph("Dataset Overview", h2_s))
    ovr_hdr = [
        Paragraph("<b>Dataset</b>",    hdr_s),
        Paragraph("<b>Rows</b>",       hdr_s),
        Paragraph("<b>Cols</b>",       hdr_s),
        Paragraph("<b>Players</b>",    hdr_s),
        Paragraph("<b>Date Range</b>", hdr_s),
        Paragraph("<b>Granularity</b>",hdr_s),
    ]
    ovr_rows = [
        ["Readiness_Data.xlsx", "14,359", "24", "28", "2024-07-01 → 2026-02-17", "Daily (player-day)"],
        ["Raw_Data.xlsx",        "9,968", "38", "84", "2024-05-02 → 2026-03-01", "Session-level"],
        ["Sessions.xlsx",        "1,206",  "8", "—",  "2024-05-02 → 2026-03-01", "Session-level (team)"],
        ["Games.xlsx",             "403",  "8", "25", "2025-07-27 → 2026-02-28", "Match-level (player)"],
    ]
    ovr_colors = [C_RD, C_RAW, C_SESS, C_GAMES]
    ovr_data = [ovr_hdr] + [
        [Paragraph(x, cell_s) for x in r] for r in ovr_rows
    ]
    ovr_t = Table(ovr_data,
                  colWidths=[1.65 * inch, 0.55 * inch, 0.45 * inch,
                              0.6 * inch, 1.65 * inch, 1.7 * inch])
    ovr_cmds = [
        ("BACKGROUND",    (0, 0), (-1, 0), C_NAVY),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING",   (0, 0), (-1, -1), 5),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 5),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    for i, c in enumerate(ovr_colors, start=1):
        ovr_cmds += [
            ("BACKGROUND", (0, i), (0, i), c),
            ("TEXTCOLOR",  (0, i), (0, i), colors.white),
            ("FONTNAME",   (0, i), (0, i), "Helvetica-Bold"),
            ("BACKGROUND", (1, i), (-1, i),
             colors.HexColor("#f8f8f8") if i % 2 == 0 else colors.white),
        ]
    ovr_t.setStyle(TableStyle(ovr_cmds))
    el.append(ovr_t)
    el.append(Paragraph(
        "<b>Note on player counts:</b> Readiness_Data and Raw_Data share 28 players; "
        "Raw_Data has 84 total players. Games has 25 players; 23 overlap with Readiness_Data.",
        tiny_s,
    ))

    # ================================================================
    # DATASET 1: Readiness_Data
    # ================================================================
    el.append(HRFlowable(width="100%", thickness=1.5, color=C_RD, spaceAfter=4))
    el.append(Paragraph(
        "1. Readiness_Data.xlsx",
        ParagraphStyle("D1", parent=h2_s, textColor=C_RD),
    ))
    el.append(Paragraph(
        "Primary dataset. Daily player monitoring data (one row per player per day) "
        "collected by OH Leuven&#39;s sports science staff. Covers 28 first-team players "
        "from 2024-07-01 to 2026-02-17 (597 days). This is the base dataset for the "
        "preprocessing pipeline — all other datasets are merged into it.",
        note_s,
    ))
    rd_cols = [
        ("Date",               "datetime", "Date of observation (daily)"),
        ("Playerkey",          "string",   "Anonymized player identifier (hash string)"),
        ("POS",                "string",   "Playing position (CD, ST, CDM, CAM, FB, WG, WB)"),
        ("MA%",                "string %", "Medical availability (%) over the last 14 days"),
        ("Att%",               "string %", "Club attendance (%) over the last 14 days"),
        ("TD",                 "float",    "Total Distance ACWR (7-day EMA / 42-day EMA ratio)"),
        ("HSD",                "float",    "High-Speed Distance (>19.8 km/h) ACWR"),
        ("Dec >3ms\u00b2",     "float",    "High deceleration (>3 m/s\u00b2) count ACWR"),
        ("Sprints",            "float",    "Sprint count (>25.2 km/h) ACWR"),
        ("Reason",             "string",   "Activity type for current day (Training, Game, Rehab, Free, etc.)"),
        ("Comment",            "string",   "Free-text coaching or medical note for current day"),
        ("TD%",                "float",    "Total distance as % of player's personal match benchmark"),
        ("HSD%",               "float",    "High-speed distance as % of personal match benchmark"),
        ("Dec >3ms\u00b2%",    "float",    "Deceleration count as % of personal match benchmark"),
        ("Sprints%",           "float",    "Sprint count as % of personal match benchmark"),
        ("Max Velocity%",      "float",    "Max velocity as % of player's personal best"),
        ("rpe (z)",            "float",    "Perceived exertion (RPE) z-score (28-day rolling window per player)"),
        ("Status",             "string",   "Medical status: Available / Attention / Injured / Sick / Absent"),
        ("Fatigue (z)",        "float",    "Self-reported fatigue z-score (28-day rolling window per player)"),
        ("Readiness (z)",      "float",    "Self-reported readiness to train z-score"),
        ("Soreness (z)",       "float",    "Self-reported muscle soreness z-score"),
        ("Sleep Quality (z)",  "float",    "Self-reported sleep quality z-score"),
        ("Stress (z)",         "float",    "Self-reported stress level z-score"),
        ("Mood (z)",           "float",    "Self-reported mood z-score"),
    ]
    el.append(ds_table(rd_cols, C_RD))
    el.append(Paragraph(
        "<b>Notes:</b> ACWR columns (TD, HSD, Dec, Sprints) are pre-computed by the monitoring system. "
        "GPS % columns express each session's load as a percentage of that player's personal match "
        "benchmark (average of the player's 5 best match values). "
        "All z-scores use a 28-day rolling window individualised per player "
        "(NOT team-level normalisation).",
        small_s,
    ))

    # ================================================================
    # DATASET 2: Raw_Data
    # ================================================================
    el.append(HRFlowable(width="100%", thickness=1.5, color=C_RAW, spaceAfter=4))
    el.append(Paragraph(
        "2. Raw_Data.xlsx",
        ParagraphStyle("D2", parent=h2_s, textColor=C_RAW),
    ))
    el.append(Paragraph(
        "Session-level GPS and heart-rate data exported from the tracking system. "
        "Contains 84 players (all 28 Readiness_Data players plus 56 others from the "
        "broader squad). Multiple rows can exist per player-day (one row per drill or "
        "session segment). In preprocessing, rows are aggregated per player-day "
        "(sum for volume metrics; weighted mean by total_minutes for HR metrics), "
        "filtered to the 28 Readiness_Data players, and shifted +1 day to become "
        "\"yesterday\" data in RTT.xlsx.",
        note_s,
    ))
    raw_cols = [
        ("Date_Value",                    "datetime", "Session date"),
        ("start_date_time",               "datetime", "Session start timestamp"),
        ("playerkey",                     "string",   "Player identifier (same hashes as Readiness_Data)"),
        ("teamkey",                       "string",   "Team identifier"),
        ("sessiontitle",                  "string",   "Session name/title"),
        ("drill_title",                   "string",   "Drill or segment title within the session"),
        ("Reason",                        "string",   "Activity reason (Training, Game, Recovery, etc.)"),
        ("Comment",                       "string",   "Free-text note"),
        ("Detail",                        "string",   "Additional detail on the session"),
        ("total_game_minutes",            "float",    "Minutes played in match (0 for non-match sessions)"),
        ("total_minutes",                 "float",    "Total session/drill duration in minutes"),
        ("total_player_load",             "float",    "Session Player Load (composite GPS exertion score)"),
        ("total_distance",                "float",    "Total distance covered in metres"),
        ("high_speed_distance",           "float",    "High-speed distance (metres, >19.8 km/h)"),
        ("distance_zone4",                "float",    "Distance in speed zone 4 (metres)"),
        ("distance_zone5",                "float",    "Distance in speed zone 5 (metres)"),
        ("high_speed_runs",               "int",      "Count of high-speed running efforts"),
        ("very_high_speed_runs",          "int",      "Count of very high-speed running efforts"),
        ("accelerations_zone4",           "int",      "Count of accelerations in zone 4 (>3 m/s\u00b2)"),
        ("decelerations_zone4",           "int",      "Count of decelerations in zone 4 (>3 m/s\u00b2)"),
        ("max_speed",                     "float",    "Maximum speed reached in the session (km/h)"),
        ("high_metabolic_load_distance",  "float",    "Distance at high metabolic load (metres)"),
        ("RPE",                           "float",    "Raw perceived exertion rating (Borg scale)"),
        ("stress_level",                  "float",    "Self-reported stress level (raw score)"),
        ("mood",                          "float",    "Self-reported mood (raw score)"),
        ("hours_sleep",                   "float",    "Hours of sleep reported"),
        ("sleep_quality",                 "float",    "Self-reported sleep quality (raw score)"),
        ("readiness",                     "float",    "Self-reported readiness to train (raw score)"),
        ("muscle_soreness",               "float",    "Self-reported muscle soreness (raw score)"),
        ("avg_heartrate",                 "float",    "Average heart rate during session (bpm)"),
        ("heart_rate_exertion",           "float",    "Heart rate exertion index (composite HR load score)"),
        ("max_heartrate",                 "float",    "Maximum heart rate during session (bpm)"),
        ("time_in_heartrate_zone1",       "float",    "Minutes in HR zone 1 (lowest intensity, <60% HRmax)"),
        ("time_in_heartrate_zone2",       "float",    "Minutes in HR zone 2 (60–70% HRmax)"),
        ("time_in_heartrate_zone3",       "float",    "Minutes in HR zone 3 (70–80% HRmax)"),
        ("time_in_heartrate_zone4",       "float",    "Minutes in HR zone 4 (80–90% HRmax)"),
        ("time_in_heartrate_zone5",       "float",    "Minutes in HR zone 5 (90–95% HRmax)"),
        ("time_in_heartrate_zone6",       "float",    "Minutes in HR zone 6 (>95% HRmax, highest intensity)"),
    ]
    el.append(ds_table(raw_cols, C_RAW))
    el.append(Paragraph(
        "<b>Columns merged into RTT.xlsx</b> (shifted +1 day): "
        "total_minutes → \"Total Minutes Yesterday\", "
        "total_distance → \"Total Distance (m) Yesterday\", "
        "high_speed_distance → \"High Speed Distance (m) Yesterday\", "
        "avg_heartrate → \"Avg Heart Rate Yesterday\", "
        "heart_rate_exertion → \"Heart Rate Exertion Yesterday\".",
        small_s,
    ))

    # ================================================================
    # DATASET 3: Sessions
    # ================================================================
    el.append(HRFlowable(width="100%", thickness=1.5, color=C_SESS, spaceAfter=4))
    el.append(Paragraph(
        "3. Sessions.xlsx",
        ParagraphStyle("D3", parent=h2_s, textColor=C_SESS),
    ))
    el.append(Paragraph(
        "Team-level session metadata (no player-level data). One row per session — "
        "gives a complete log of all training sessions and matches in the monitoring system. "
        "Used in preprocessing only to identify match days for the Match Day feature in RTT.xlsx.",
        note_s,
    ))
    sess_cols = [
        ("Date_Value",      "datetime", "Session date"),
        ("teamkey",         "string",   "Team identifier"),
        ("matchday",        "string",   "Match-day code: \"MD 0\" = match day, \"MD -1\" = day before, etc."),
        ("start_date_time", "datetime", "Session start timestamp"),
        ("session_title",   "string",   "Session name (e.g., \"Training\", \"MD 0\", \"Recovery\")"),
        ("session_type",    "string",   "High-level session type classification"),
        ("workout_type",    "string",   "Workout classification"),
        ("Reason",          "string",   "Activity reason"),
    ]
    el.append(ds_table(sess_cols, C_SESS))
    el.append(Paragraph(
        "<b>Preprocessing usage:</b> Rows where matchday = \"MD 0\" identify OHL first-team match days. "
        "These dates feed into the Match Day column in RTT.xlsx (team-level: 1 = match today, 0 = no match). "
        "No other columns from Sessions.xlsx are used.",
        small_s,
    ))

    # ================================================================
    # DATASET 4: Games
    # ================================================================
    el.append(HRFlowable(width="100%", thickness=1.5, color=C_GAMES, spaceAfter=4))
    el.append(Paragraph(
        "4. Games.xlsx",
        ParagraphStyle("D4", parent=h2_s, textColor=C_GAMES),
    ))
    el.append(Paragraph(
        "Match-level physical performance data per player. Contains 403 match appearances "
        "across 25 players (23 overlapping with Readiness_Data). Metrics are normalised "
        "per Ball-In-Play (BIP) minute to allow fair comparison across matches of different "
        "durations and playing time. In preprocessing, match data is shifted +1 day so "
        "a match on date t becomes \"yesterday\" match data in the row for date t+1.",
        note_s,
    ))
    games_cols = [
        ("Team",                       "string",   "Team name"),
        ("date",                       "datetime", "Match date"),
        ("match_week",                 "int",      "Match week number in the season"),
        ("Game",                       "string",   "Match description (e.g., \"OHL vs Antwerp — JPL R14\")"),
        ("High Intensity Per BIP (m)", "float",    "High-intensity running distance per BIP minute (m). "
                                                   "Primary match performance metric and causal outcome Y."),
        ("HIT Efforts per BIP",        "float",    "High-intensity effort count per Ball-In-Play minute"),
        ("minutes_played",             "float",    "Total minutes played in the match"),
        ("playernames.playerkey",      "string",   "Hashed player identifier (same hashes as Readiness_Data)"),
    ]
    el.append(ds_table(games_cols, C_GAMES))
    el.append(Paragraph(
        "<b>Columns merged into RTT.xlsx</b> (shifted +1 day): "
        "High Intensity Per BIP → \"Match High Intensity Per BIP Yesterday\", "
        "HIT Efforts per BIP → \"Match HIT Efforts Per BIP Yesterday\", "
        "minutes_played → \"Match Minutes Played Yesterday\". "
        "These columns are NaN on approximately 68% of rows (all non-match-following days). "
        "<b>High Intensity Per BIP is the causal outcome (Y)</b> in the DTR framework.",
        small_s,
    ))

    # ================================================================
    # Definitions
    # ================================================================
    el.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cccccc"), spaceAfter=4))
    el.append(Paragraph("Definitions", h2_s))
    for d in [
        ("<b>ACWR (Acute:Chronic Workload Ratio)</b> — Ratio of acute (7-day EMA) to chronic "
         "(42-day EMA) training load. Sweet spot 0.8–1.3; values >1.5 indicate elevated injury "
         "risk (Gabbett, 2016)."),
        ("<b>EMA (Exponential Moving Average)</b> — Weighted average giving more weight to recent "
         "observations. More responsive than a simple rolling mean."),
        ("<b>Personal Benchmark</b> — Average of a player's top 5 match performances for each GPS "
         "metric. GPS % columns are computed relative to this individual benchmark."),
        ("<b>Z-score</b> — Standardized score: 0 = player's own mean, ±1 = one standard deviation. "
         "Z-scores use a 28-day rolling window, individualised per player."),
        ("<b>BIP (Ball In Play)</b> — Time during a match when the ball is actively in play. "
         "Normalising by BIP minutes enables fair cross-match performance comparisons."),
        ("<b>Player Load</b> — Composite GPS-derived exertion score weighting accelerations, "
         "decelerations, and changes of direction."),
        ("<b>HR Zones 1–6</b> — Heart rate intensity zones defined as percentages of maximum "
         "heart rate (e.g., Zone 1: <60%, Zone 6: >95%)."),
        ("<b>DTR (Dynamic Treatment Regime)</b> — A sequence of decision rules mapping the "
         "evolving player state and treatment history to an optimal training intensity at each "
         "decision point. The causal goal is to maximise High Intensity Per BIP."),
    ]:
        el.append(Paragraph(d, def_s))

    doc.build(el)
    print(f"Saved to: {OUT}")
    sz = OUT.stat().st_size / 1024
    print(f"File size: {sz:.0f} KB")


if __name__ == "__main__":
    build_pdf()
