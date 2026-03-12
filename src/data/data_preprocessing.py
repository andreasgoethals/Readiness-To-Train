"""
Data Preprocessing Pipeline — Multi-Dataset Integration
Combines Readiness_Data, Raw_Data, Sessions, and Games into a single
analysis-ready dataset (RTT.xlsx) with feature engineering.

This module loads raw xlsx files from the OHL player monitoring system, merges
them by player ID and date, applies temporal shifting (so training/match data
from day t becomes "yesterday" data in the row for day t+1), engineers features,
and saves the result as an Excel workbook plus a PDF data dictionary.

Pipeline Summary (executed in order by preprocess_data()):
    1.  Load all raw xlsx files (Readiness_Data, Raw_Data, Sessions, Games)
    2.  Map complex Playerkey strings -> simple sequential Player IDs (1-28)
    3.  Rename cryptic column abbreviations -> human-readable names
    4.  Clean percentage columns (string -> integer conversion)
    5.  Categorize free-text comments into keyword-based categories
    6.  Merge Raw_Data columns (total_minutes, total_distance, high_speed_distance,
        avg_heartrate, heart_rate_exertion) — shifted so day-of -> yesterday
    7.  Merge Games columns (High Intensity Per BIP, HIT Efforts Per BIP,
        minutes_played) — shifted so match-day -> day after match
    8.  Create composite wellness scores (Physical State, Mental State, Overall Wellbeing)
    9.  Create temporal features (Activity Type Today, Days Since Game, Days Until Match)
    10. Create match day features (Match Day team-level, Selected player-level)
    11. Detect status decrease transitions
    12. Flag dangerous ACWR values (>1.5)
    13. Create Training Intensity Yesterday composite (tanh(mean(TD%, HSD%, Dec%, Sprints%) / 100), range [0, 1))
    14. Cap GPS benchmark % variables at a configurable ceiling (default 250%)
    15. Filter dataset to a configurable date range (default 2025-07-27 -> 2026-02-28)
    16. Reorder columns into temporal groups
    17. Save RTT.xlsx + auto-generate PDF data dictionary

=== TEMPORAL SEMANTICS ===

Each row represents one player-day (date t). Within that row, variables have
different temporal positions:

    BEFORE day t:   Activity Type Yesterday, GPS %, ACWR, Comments, RPE,
                    Raw_Data metrics (shifted: recorded on t-1),
                    Games metrics (shifted: match played on t-1)

    MORNING of t:   Wellness z-scores, Status, Days Since Game, Days Until Match

    AFTER morning:  Activity Type Today
                    (session type assigned after morning assessment;
                    derived from the NEXT row's Activity Type Yesterday)

=== DATA SOURCES ===

Readiness_Data.xlsx   — Base dataset. ALL rows retained before date filter.
Raw_Data.xlsx         — Session-level GPS/HR. Filtered to RD players & dates,
                        aggregated per player-day (sum for load, weighted mean
                        for HR), then shifted forward 1 day.
Sessions.xlsx         — Team-level session metadata. Loaded for reference.
Games.xlsx            — Match performance data. Filtered to RD players & dates,
                        shifted forward 1 day (match data -> next row).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon


# =============================================================================
# DATA LOADING
# =============================================================================

def get_project_root():
    """Get the project root directory (two levels up from src/data/)."""
    return Path(__file__).parent.parent.parent


def load_all_raw_data():
    """
    Load all raw xlsx datasets from data/raw/.

    Returns
    -------
    dict
        Keys: 'rd1', 'raw', 'sessions', 'games' — each a pd.DataFrame.
    """
    raw_dir = get_project_root() / "data" / "raw"

    print("Loading Readiness_Data.xlsx...")
    rd1 = pd.read_excel(raw_dir / "Readiness_Data.xlsx", engine='openpyxl')
    rd1['Date'] = pd.to_datetime(rd1['Date'])
    print(f"  {rd1.shape[0]:,} rows x {rd1.shape[1]} columns, "
          f"{rd1['Playerkey'].nunique()} players")

    print("Loading Raw_Data.xlsx...")
    raw = pd.read_excel(raw_dir / "Raw_Data.xlsx", engine='openpyxl')
    raw['Date_Value'] = pd.to_datetime(raw['Date_Value'])
    print(f"  -> {raw.shape[0]:,} rows x {raw.shape[1]} columns, "
          f"{raw['playerkey'].nunique()} players")

    print("Loading Sessions.xlsx...")
    sessions = pd.read_excel(raw_dir / "Sessions.xlsx", engine='openpyxl')
    sessions['Date_Value'] = pd.to_datetime(sessions['Date_Value'])
    print(f"  -> {sessions.shape[0]:,} rows x {sessions.shape[1]} columns")

    print("Loading Games.xlsx...")
    games = pd.read_excel(raw_dir / "Games.xlsx", engine='openpyxl')
    games['date'] = pd.to_datetime(games['date'])
    print(f"  -> {games.shape[0]:,} rows x {games.shape[1]} columns, "
          f"{games['playernames.playerkey'].nunique()} players")

    return {'rd1': rd1, 'raw': raw, 'sessions': sessions, 'games': games}


# =============================================================================
# BASIC TRANSFORMATIONS (applied to Readiness_Data base)
# =============================================================================

def map_player_ids(df):
    """
    Convert complex player keys to simple sequential IDs (1, 2, 3, ...).

    The mapping is deterministic: players are sorted alphabetically by
    their Playerkey, then assigned IDs starting from 1.

    Returns
    -------
    tuple of (pd.DataFrame, dict)
        - DataFrame with new 'Player ID' column added
        - Dictionary mapping original Playerkey -> new Player ID
    """
    print("\nMapping player IDs...")
    unique_players = sorted(df['Playerkey'].unique())
    player_mapping = {old_key: idx + 1 for idx, old_key in enumerate(unique_players)}
    df['Player ID'] = df['Playerkey'].map(player_mapping)
    n = len(unique_players)
    print(f"  Mapped {n} players to IDs 1-{n}")
    return df, player_mapping


def rename_columns(df):
    """
    Rename cryptic column abbreviations to human-readable names.

    The 'Yesterday' suffix indicates these metrics were recorded on the day
    BEFORE the current observation row (t-1 data).
    """
    print("\nRenaming columns...")
    rename_dict = {
        'MA%': 'Medical Availability Last 14 Days',
        'Att%': 'Club Attendance Last 14 Days',
        'POS': 'Position',
        'TD': 'Total Distance (ACWR) Yesterday',
        'HSD': 'High Speed Distance (ACWR) Yesterday',
        'Dec >3ms²': 'High Decelerations (ACWR) Yesterday',
        'Sprints': 'Sprints (ACWR) Yesterday',
        'Reason': 'Activity Type Yesterday',
        'Comment': 'Comment Yesterday',
        'TD%': 'Total Distance % Yesterday',
        'HSD%': 'High Speed Distance % Yesterday',
        'Dec >3ms²%': 'High Decelerations % Yesterday',
        'Sprints%': 'Sprints % Yesterday',
        'Max, Velocity%': 'Max Velocity % Yesterday',
        'rpe (z)': 'Perceived Exertion Yesterday'
    }
    df = df.rename(columns=rename_dict)
    print(f"  Renamed {len(rename_dict)} columns")
    return df


def clean_percentage_columns(df):
    """Convert percentage columns from strings to integers."""
    print("\nCleaning percentage columns...")
    percentage_cols = ['Medical Availability Last 14 Days', 'Club Attendance Last 14 Days']
    for col in percentage_cols:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: int(float(x)) if pd.notna(x) and x != '' else x
            )
            print(f"  {col}: converted to integer type")
    return df


# =============================================================================
# COMMENT CATEGORIZATION
# =============================================================================

def categorize_comment(comment):
    """
    Categorize a single free-text comment into a predefined category.

    Returns one of: 'recovery', 'discomfort', 'stiffness', 'sick', 'hydrops',
    'cramp', 'empty', or 'other'.
    """
    if pd.isna(comment) or comment == '':
        return 'empty'

    comment_lower = str(comment).lower()

    categories = {
        'recovery': ['recovery'],
        'discomfort': ['discomfort'],
        'stiffness': ['stiffness'],
        'sick': ['viral', 'infection', 'symptoms', 'cold'],
        'hydrops': ['hydrops'],
        'cramp': ['cramp']
    }

    matches = []
    for category, keywords in categories.items():
        if any(keyword in comment_lower for keyword in keywords):
            matches.append(category)

    if len(matches) == 0:
        return 'other'
    elif len(matches) == 1:
        return matches[0]
    else:
        return 'other'


def add_comment_category(df):
    """Add 'Comment Category Yesterday' column by categorizing free-text comments."""
    print("\nAdding Comment Category Yesterday...")
    df['Comment Category Yesterday'] = df['Comment Yesterday'].apply(categorize_comment)
    category_counts = df['Comment Category Yesterday'].value_counts()
    print(f"  Added Comment Category Yesterday:")
    for category, count in category_counts.items():
        print(f"    - {category}: {count}")
    return df


# =============================================================================
# EXTERNAL DATA MERGING
# =============================================================================

def _european_to_float(series):
    """Convert European decimal format ('32,14') to float."""
    return pd.to_numeric(
        series.astype(str).str.replace(',', '.', regex=False),
        errors='coerce'
    )


def merge_raw_data(df, raw, player_mapping):
    """
    Merge GPS and heart rate data from Raw_Data.xlsx into the base dataset.

    Raw_Data records session-level data ON THE DAY ITSELF. In our dataset's
    temporal structure, this should appear as "yesterday" data. So we shift:
    Raw_Data from date t -> appears in our dataset's row at date t+1.

    When a player has multiple sessions on the same day (e.g., morning training +
    afternoon session), we aggregate:
    - SUM for volume metrics: total_minutes, total_distance, high_speed_distance
    - Weighted MEAN (by total_minutes) for intensity: avg_heartrate, heart_rate_exertion
    """
    print("\nMerging Raw_Data (GPS/HR -> shifted to yesterday)...")

    rd1_players = set(df['Playerkey'].unique())
    raw_f = raw[raw['playerkey'].isin(rd1_players)].copy()
    print(f"  Filtered Raw_Data: {len(raw_f):,} rows for {raw_f['playerkey'].nunique()} players")

    # Convert heart_rate_exertion from European decimal string to float
    raw_f['heart_rate_exertion'] = _european_to_float(raw_f['heart_rate_exertion'])

    # Ensure numeric types
    for col in ['total_minutes', 'total_distance', 'high_speed_distance', 'avg_heartrate']:
        raw_f[col] = pd.to_numeric(raw_f[col], errors='coerce')

    # Aggregate per player-day
    def agg_player_day(group):
        result = {}
        result['total_minutes'] = group['total_minutes'].sum()
        result['total_distance'] = group['total_distance'].sum()
        result['high_speed_distance'] = group['high_speed_distance'].sum()
        weights = group['total_minutes'].fillna(0)
        total_weight = weights.sum()
        if total_weight > 0:
            result['avg_heartrate'] = np.average(
                group['avg_heartrate'].fillna(0), weights=weights
            )
            result['heart_rate_exertion'] = np.average(
                group['heart_rate_exertion'].fillna(0), weights=weights
            )
        else:
            result['avg_heartrate'] = group['avg_heartrate'].mean()
            result['heart_rate_exertion'] = group['heart_rate_exertion'].mean()
        return pd.Series(result)

    raw_agg = raw_f.groupby(['playerkey', 'Date_Value']).apply(
        agg_player_day, include_groups=False
    ).reset_index()
    print(f"  Aggregated to {len(raw_agg):,} player-day observations")

    raw_agg = raw_agg.rename(columns={
        'playerkey': 'Playerkey',
        'Date_Value': '_raw_date',
        'total_minutes': 'Total Minutes Yesterday',
        'total_distance': 'Total Distance (m) Yesterday',
        'high_speed_distance': 'High Speed Distance (m) Yesterday',
        'avg_heartrate': 'Avg Heart Rate Yesterday',
        'heart_rate_exertion': 'Heart Rate Exertion Yesterday',
    })

    # Shift: Raw_Data from date t -> appears in row at date t+1
    df['_merge_date'] = df['Date'] - pd.Timedelta(days=1)

    df = df.merge(
        raw_agg,
        left_on=['Playerkey', '_merge_date'],
        right_on=['Playerkey', '_raw_date'],
        how='left'
    )

    df = df.drop(columns=['_merge_date', '_raw_date'], errors='ignore')

    for col in ['Total Minutes Yesterday', 'Total Distance (m) Yesterday',
                'High Speed Distance (m) Yesterday', 'Avg Heart Rate Yesterday',
                'Heart Rate Exertion Yesterday']:
        n_valid = df[col].notna().sum()
        pct = n_valid / len(df) * 100
        print(f"  {col}: {n_valid:,} valid ({pct:.1f}%)")

    return df


def merge_games_data(df, games):
    """
    Merge match performance data from Games.xlsx into the base dataset.

    Games.xlsx records match performance ON THE MATCH DAY. Match data from
    date t should appear as "yesterday" data in the row at date t+1.
    These columns are only filled on the day AFTER a match.
    """
    print("\nMerging Games data (match performance -> shifted to yesterday)...")

    rd1_players = set(df['Playerkey'].unique())
    games_f = games[games['playernames.playerkey'].isin(rd1_players)].copy()
    print(f"  Filtered Games: {len(games_f):,} rows for "
          f"{games_f['playernames.playerkey'].nunique()} players")

    games_merge = games_f[['playernames.playerkey', 'date',
                            'High Intensity Per BIP (m)', 'HIT Efforts per BIP',
                            'minutes_played']].copy()
    games_merge = games_merge.rename(columns={
        'playernames.playerkey': 'Playerkey',
        'date': '_game_date',
        'High Intensity Per BIP (m)': 'Match HID Per BIP Yesterday',
        'HIT Efforts per BIP': 'Match HIE Per BIP Yesterday',
        'minutes_played': 'Match Minutes Played Yesterday',
    })

    # Shift: game on date t -> appears in row at date t+1
    df['_merge_date'] = df['Date'] - pd.Timedelta(days=1)

    df = df.merge(
        games_merge,
        left_on=['Playerkey', '_merge_date'],
        right_on=['Playerkey', '_game_date'],
        how='left'
    )

    df = df.drop(columns=['_merge_date', '_game_date'], errors='ignore')

    for col in ['Match HID Per BIP Yesterday',
                'Match HIE Per BIP Yesterday',
                'Match Minutes Played Yesterday']:
        n_valid = df[col].notna().sum()
        print(f"  {col}: {n_valid:,} valid entries")

    # Composite outcome: geometric mean of HID and HIE per BIP, weighted by
    # sqrt(clip(minutes_played, 15, 90) / 90)  — playing time is clamped to
    # [15, 90]: the floor of 15 prevents very brief cameo appearances from being
    # penalised too harshly; the ceiling of 90 gives full credit (weight=1) to
    # anyone who played a full match.
    df['Match Intensity Yesterday'] = (
        np.sqrt(df['Match HID Per BIP Yesterday'] * df['Match HIE Per BIP Yesterday'])
        * np.sqrt(np.clip(df['Match Minutes Played Yesterday'], 15, 90) / 90)
    )
    n_valid_mi = df['Match Intensity Yesterday'].notna().sum()
    print(f"  Match Intensity Yesterday: {n_valid_mi:,} valid entries")

    return df


# =============================================================================
# COMPOSITE SCORES
# =============================================================================

def add_physical_state(df):
    """Add 'Physical State' — mean of Fatigue, Readiness, Soreness z-scores."""
    print("\nAdding Physical State...")
    physical_cols = ['Fatigue (z)', 'Readiness (z)', 'Soreness (z)']
    df['Physical State'] = df[physical_cols].mean(axis=1)
    n_valid = df['Physical State'].notna().sum()
    print(f"  Added Physical State ({n_valid:,} valid entries)")
    return df


def add_mental_state(df):
    """Add 'Mental State' — mean of Sleep Quality, Stress, Mood z-scores."""
    print("\nAdding Mental State...")
    mental_cols = ['Sleep Quality (z)', 'Stress (z)', 'Mood (z)']
    df['Mental State'] = df[mental_cols].mean(axis=1)
    n_valid = df['Mental State'].notna().sum()
    print(f"  Added Mental State ({n_valid:,} valid entries)")
    return df


def add_overall_wellbeing(df):
    """Add 'Overall Wellbeing' — mean of all 6 wellness z-scores."""
    print("\nAdding Overall Wellbeing...")
    wellbeing_cols = [
        'Fatigue (z)', 'Readiness (z)', 'Soreness (z)',
        'Sleep Quality (z)', 'Stress (z)', 'Mood (z)'
    ]
    df['Overall Wellbeing'] = df[wellbeing_cols].mean(axis=1)
    n_valid = df['Overall Wellbeing'].notna().sum()
    print(f"  Added Overall Wellbeing ({n_valid:,} valid entries)")
    return df


# =============================================================================
# TEMPORAL FEATURES
# =============================================================================

def add_activity_type_today(df):
    """
    Add 'Activity Type Today' — session type assigned on day t, AFTER morning assessment.

    Derived from the NEXT row's 'Activity Type Yesterday' for the same player.
    Only valid when the next row is exactly the next calendar day (no date gaps).
    """
    print("\nAdding Activity Type Today...")
    df_sorted = df.sort_values(['Player ID', 'Date']).copy()

    df_sorted['Activity Type Today'] = df_sorted.groupby('Player ID')[
        'Activity Type Yesterday'
    ].shift(-1)

    # Date continuity guard
    df_sorted['_next_date'] = df_sorted.groupby('Player ID')['Date'].shift(-1)
    gap_mask = df_sorted['_next_date'] != (df_sorted['Date'] + pd.Timedelta(days=1))
    df_sorted.loc[gap_mask, 'Activity Type Today'] = np.nan
    df_sorted = df_sorted.drop(columns=['_next_date'])

    # Map back to original order
    df_sorted['_tk'] = df_sorted['Player ID'].astype(str) + '_' + df_sorted['Date'].astype(str)
    df['_tk'] = df['Player ID'].astype(str) + '_' + df['Date'].astype(str)
    mapping = df_sorted.set_index('_tk')['Activity Type Today'].to_dict()
    df['Activity Type Today'] = df['_tk'].map(mapping)
    df = df.drop(columns=['_tk'])

    n_valid = df['Activity Type Today'].notna().sum()
    print(f"  Added Activity Type Today ({n_valid:,} valid entries)")
    return df


def add_match_day_and_selected(df):
    """
    Add 'Match Day' (team-level binary) and 'Selected' (player-level).

    Match Day = 1 if ANY player has Activity Type Today == 'Game' on that date.
    Selected  = 1 if THIS player's Activity Type Today == 'Game', 0 otherwise,
                NaN on non-match days.

    Only exact 'Game' counts — excludes youth/national team games.
    """
    print("\nAdding Match Day and Selected...")

    game_mask = df['Activity Type Today'].str.strip().str.lower() == 'game'
    match_dates = set(df.loc[game_mask, 'Date'].unique())
    df['Match Day'] = df['Date'].apply(lambda d: 1 if d in match_dates else 0)

    def assign_selected(row):
        if row['Match Day'] == 0:
            return np.nan
        activity = row['Activity Type Today']
        if pd.isna(activity):
            return np.nan
        if str(activity).strip().lower() == 'game':
            return 1
        return 0

    df['Selected'] = df.apply(assign_selected, axis=1)

    n_match_days = len(match_dates)
    n_selected = int(df['Selected'].sum()) if df['Selected'].notna().any() else 0
    n_not_selected = int((df['Selected'] == 0).sum())
    print(f"  Added Match Day ({n_match_days} match days identified)")
    print(f"  Added Selected ({n_selected} selections, {n_not_selected} non-selections on match days)")
    return df


def add_days_since_game(df):
    """
    Add 'Days Since Game' — calendar days since last COMPLETED match (min=1, never 0).

    Measured in the morning. On match day the game hasn't occurred yet, so the
    count refers to the previous match. Only exact 'Game' in Activity Type
    Yesterday counts (excludes youth/national team).
    """
    print("\nAdding Days Since Game...")
    df_sorted = df.sort_values(['Player ID', 'Date']).copy()

    def calc_days_since_game(group):
        is_game_yesterday = group['Activity Type Yesterday'].str.strip().str.lower() == 'game'
        actual_game_dates = group.loc[is_game_yesterday, 'Date'] - pd.Timedelta(days=1)
        result = pd.Series(index=group.index, dtype=float)
        for idx, row in group.iterrows():
            current_date = row['Date']
            prior_games = actual_game_dates[actual_game_dates < current_date]
            if len(prior_games) > 0:
                result[idx] = (current_date - prior_games.max()).days
            else:
                result[idx] = np.nan
        return result

    df_sorted['Days Since Game'] = df_sorted.groupby(
        'Player ID', group_keys=False
    ).apply(calc_days_since_game, include_groups=False)

    # Map back
    df_sorted['_tk'] = df_sorted['Player ID'].astype(str) + '_' + df_sorted['Date'].astype(str)
    df['_tk'] = df['Player ID'].astype(str) + '_' + df['Date'].astype(str)
    mapping = df_sorted.set_index('_tk')['Days Since Game'].to_dict()
    df['Days Since Game'] = df['_tk'].map(mapping)
    df = df.drop(columns=['_tk'])

    n_valid = df['Days Since Game'].notna().sum()
    n_zeros = (df['Days Since Game'] == 0).sum()
    print(f"  Added Days Since Game ({n_valid:,} valid entries, {n_zeros} zeros [should be 0])")
    if n_zeros > 0:
        print(f"  WARNING: Found {n_zeros} zero values!")
    return df


def add_days_until_match(df):
    """
    Add 'Days Until Match' — calendar days until next scheduled match (0 on match day).

    Derived from Activity Type Today. Only exact 'Game' counts.
    """
    print("\nAdding Days Until Match...")
    df_sorted = df.sort_values(['Player ID', 'Date']).copy()

    def calc_days_until_match(group):
        is_game_today = group['Activity Type Today'].str.strip().str.lower() == 'game'
        game_dates = group.loc[is_game_today, 'Date']
        result = pd.Series(index=group.index, dtype=float)
        for idx, row in group.iterrows():
            current_date = row['Date']
            future_games = game_dates[game_dates >= current_date]
            if len(future_games) > 0:
                result[idx] = (future_games.min() - current_date).days
            else:
                result[idx] = np.nan
        return result

    df_sorted['Days Until Match'] = df_sorted.groupby(
        'Player ID', group_keys=False
    ).apply(calc_days_until_match, include_groups=False)

    # Map back
    df_sorted['_tk'] = df_sorted['Player ID'].astype(str) + '_' + df_sorted['Date'].astype(str)
    df['_tk'] = df['Player ID'].astype(str) + '_' + df['Date'].astype(str)
    mapping = df_sorted.set_index('_tk')['Days Until Match'].to_dict()
    df['Days Until Match'] = df['_tk'].map(mapping)
    df = df.drop(columns=['_tk'])

    n_valid = df['Days Until Match'].notna().sum()
    print(f"  Added Days Until Match ({n_valid:,} valid entries)")
    return df


# =============================================================================
# STATUS FEATURES
# =============================================================================

def add_status_decrease(df):
    """
    Add binary flag for status decrease — primary prediction target.

    A status decrease is flagged (1) for these transitions:
        Available -> Attention, Available -> Injured, Attention -> Injured
    """
    print("\nAdding Status Decrease...")
    decrease_transitions = {
        ('available', 'attention'),
        ('available', 'injured'),
        ('attention', 'injured')
    }

    df_sorted = df.sort_values(['Player ID', 'Date']).copy()
    df_sorted['Previous Status'] = df_sorted.groupby('Player ID')['Status'].shift(1)

    def is_decrease(row):
        if pd.isna(row['Previous Status']) or pd.isna(row['Status']):
            return 0
        prev = str(row['Previous Status']).lower()
        curr = str(row['Status']).lower()
        return 1 if (prev, curr) in decrease_transitions else 0

    df_sorted['Status Decrease'] = df_sorted.apply(is_decrease, axis=1)

    # Map back
    df_sorted['_tk'] = df_sorted['Player ID'].astype(str) + '_' + df_sorted['Date'].astype(str)
    df['_tk'] = df['Player ID'].astype(str) + '_' + df['Date'].astype(str)
    mapping = df_sorted.set_index('_tk')['Status Decrease'].to_dict()
    df['Status Decrease'] = df['_tk'].map(mapping)
    df = df.drop(columns=['_tk'])

    n_decrease = df['Status Decrease'].sum()
    print(f"  Added Status Decrease ({int(n_decrease):,} status decreases detected)")
    return df


# =============================================================================
# ACWR DANGER ZONE
# =============================================================================

def add_any_acwr_danger(df):
    """
    Add binary flag if ANY ACWR metric exceeds 1.5 (Gabbett 2016 danger zone).

    Checks Total Distance, High Speed Distance, High Decelerations, Sprints ACWRs.
    NaN ACWRs -> NaN result (unknown, not safe).
    """
    print("\nAdding Any ACWR Danger...")
    acwr_cols = [
        'Total Distance (ACWR) Yesterday',
        'High Speed Distance (ACWR) Yesterday',
        'High Decelerations (ACWR) Yesterday',
        'Sprints (ACWR) Yesterday'
    ]
    acwr_cols = [c for c in acwr_cols if c in df.columns]

    acwr_df = df[acwr_cols]
    danger_flags = acwr_df.gt(1.5)
    all_nan = acwr_df.isna().all(axis=1)

    df['Any ACWR Danger'] = danger_flags.any(axis=1).astype(int)
    df.loc[all_nan, 'Any ACWR Danger'] = np.nan

    n_danger = df['Any ACWR Danger'].sum()
    print(f"  Added Any ACWR Danger ({n_danger:,} observations in danger zone)")
    return df


# =============================================================================
# TRAINING INTENSITY COMPOSITE
# =============================================================================

def add_training_intensity_yesterday(df):
    """
    Add 'Training Intensity Yesterday' — composite GPS load score in [0, 1).

    Computed as the hyperbolic tangent (tanh) of the mean of the four GPS
    benchmark percentages divided by 100:

        tanh(mean(TD%, HSD%, Dec%, Sprints%) / 100)

    tanh maps [0, +inf) -> [0, 1) smoothly with no hard ceiling. Representative
    values:
        0%   of match benchmark  ->  0.00
        50%  of match benchmark  ->  0.46
        78%  (typical training)  ->  0.65
        100% (match-level load)  ->  0.76
        130% (hard session)      ->  0.86
        200% (very extreme)      ->  0.96

    Max Velocity % is intentionally excluded (peak-speed capacity, not load volume).

    NaN handling:
      - If ALL four GPS % columns are NaN -> NaN
      - If some are NaN, mean computed over available values (skipna=True)
    """
    print("\nAdding Training Intensity Yesterday...")
    gps_pct_cols = [
        'Total Distance % Yesterday',
        'High Speed Distance % Yesterday',
        'High Decelerations % Yesterday',
        'Sprints % Yesterday',
    ]
    present = [c for c in gps_pct_cols if c in df.columns]
    if not present:
        print("  WARNING: No GPS % columns found — skipping Training Intensity Yesterday")
        return df

    raw_mean = df[present].mean(axis=1)
    df['Training Intensity Yesterday'] = np.tanh(raw_mean / 100.0)

    valid = df['Training Intensity Yesterday'].notna()
    n_valid = valid.sum()
    n_nan = (~valid).sum()
    mean_val = df.loc[valid, 'Training Intensity Yesterday'].mean()
    median_val = df.loc[valid, 'Training Intensity Yesterday'].median()
    n_at_one = (df.loc[valid, 'Training Intensity Yesterday'] == 1.0).sum()

    print(f"  Created from {len(present)} GPS % columns (tanh soft cap)")
    print(f"  Valid: {n_valid:,}, NaN: {n_nan:,}")
    print(f"  Mean: {mean_val:.3f}, Median: {median_val:.3f}")
    if n_at_one > 0:
        print(f"  WARNING: {n_at_one:,} values equal exactly 1.0 (unexpected with tanh)")
    else:
        print(f"  No values at exactly 1.0 (as expected with tanh soft cap)")
    return df


# =============================================================================
# BENCHMARK % CAPPING
# =============================================================================

def cap_benchmark_percentages(df, cap=250):
    """
    Cap GPS benchmark % columns at a maximum value to reduce extreme outlier influence.

    Values above `cap` are clipped to `cap`. Extreme values arise when a player's
    personal benchmark is very low (e.g. early-season or injury-return matches).

    Parameters
    ----------
    df : pd.DataFrame
    cap : int or float, default 250
        Upper bound for benchmark % columns. Values above this are clipped to cap.

    Returns
    -------
    pd.DataFrame
    """
    print(f"\nCapping benchmark % columns at {cap}%...")
    benchmark_cols = [
        'Total Distance % Yesterday',
        'High Speed Distance % Yesterday',
        'High Decelerations % Yesterday',
        'Sprints % Yesterday',
        'Max Velocity % Yesterday',
    ]
    total_capped = 0
    for col in benchmark_cols:
        if col in df.columns:
            n_above = (df[col] > cap).sum()
            if n_above > 0:
                df[col] = df[col].clip(upper=cap)
                print(f"  {col}: capped {n_above} values above {cap}%")
            else:
                print(f"  {col}: no values above {cap}%")
            total_capped += n_above
    print(f"  Total values capped: {total_capped}")
    return df


# =============================================================================
# DATE RANGE FILTER
# =============================================================================

def filter_date_range(df, date_min=None, date_max=None):
    """
    Filter the dataset to a specific date range (inclusive on both ends).

    Parameters
    ----------
    df : pd.DataFrame
        Must have a 'Date' column parsed as datetime.
    date_min : str or None, e.g. '2025-07-27'
        Start date (inclusive). If None, no lower bound applied.
    date_max : str or None, e.g. '2026-02-28'
        End date (inclusive). If None, no upper bound applied.

    Returns
    -------
    pd.DataFrame
        Filtered and index-reset DataFrame.
    """
    if date_min is None and date_max is None:
        return df

    n_before = len(df)
    label = f"{date_min} \u2192 {date_max}" if date_min and date_max else (date_min or date_max)
    print(f"\nFiltering date range: {label}...")

    if date_min is not None:
        df = df[df['Date'] >= pd.Timestamp(date_min)]
    if date_max is not None:
        df = df[df['Date'] <= pd.Timestamp(date_max)]

    df = df.reset_index(drop=True)
    n_after = len(df)
    print(f"  Rows before: {n_before:,}  \u2192  after filter: {n_after:,} (removed {n_before - n_after:,})")
    return df


# =============================================================================
# COLUMN REORDERING
# =============================================================================

def reorder_columns(df):
    """
    Reorder columns into temporal groups reflecting information availability.

    GROUP 1: Identifiers (always known)
    GROUP 2: Historical context (before day t)
    GROUP 3: Yesterday's (t-1) data from Readiness_Data
    GROUP 4: Yesterday's (t-1) data from Raw_Data (GPS/HR session data)
    GROUP 5: Yesterday's (t-1) data from Games (match performance)
    GROUP 6: Today's (t) morning assessment — covariates
    GROUP 7: Today's (t) post-assessment — treatment/activity
    """
    print("\nReordering columns...")
    column_order = [
        # --- GROUP 1: Identifiers ---
        'Date', 'Playerkey', 'Player ID', 'Position',
        # --- GROUP 2: Historical Context ---
        'Medical Availability Last 14 Days',
        'Club Attendance Last 14 Days',
        # --- GROUP 3: Yesterday (t-1) from Readiness_Data ---
        'Total Distance (ACWR) Yesterday',
        'High Speed Distance (ACWR) Yesterday',
        'High Decelerations (ACWR) Yesterday',
        'Sprints (ACWR) Yesterday',
        'Any ACWR Danger',
        'Activity Type Yesterday',
        'Comment Yesterday',
        'Comment Category Yesterday',
        'Total Distance % Yesterday',
        'High Speed Distance % Yesterday',
        'High Decelerations % Yesterday',
        'Sprints % Yesterday',
        'Max Velocity % Yesterday',
        'Training Intensity Yesterday',
        'Perceived Exertion Yesterday',
        # --- GROUP 4: Yesterday (t-1) from Raw_Data (GPS/HR) ---
        'Total Minutes Yesterday',
        'Total Distance (m) Yesterday',
        'High Speed Distance (m) Yesterday',
        'Avg Heart Rate Yesterday',
        'Heart Rate Exertion Yesterday',
        # --- GROUP 5: Yesterday (t-1) from Games (match performance) ---
        'Match HID Per BIP Yesterday',
        'Match HIE Per BIP Yesterday',
        'Match Minutes Played Yesterday',
        'Match Intensity Yesterday',
        # --- GROUP 6: Today (t) morning assessment — COVARIATES ---
        'Status',
        'Status Decrease',
        'Fatigue (z)', 'Readiness (z)', 'Soreness (z)',
        'Physical State',
        'Sleep Quality (z)', 'Stress (z)', 'Mood (z)',
        'Mental State',
        'Overall Wellbeing',
        'Days Since Game',
        'Days Until Match',
        'Match Day',
        # --- GROUP 7: Today (t) post-assessment ---
        'Activity Type Today',
        'Selected',
    ]
    available = [col for col in column_order if col in df.columns]
    remaining = [col for col in df.columns if col not in available]
    final_order = available + remaining
    df = df[final_order]
    print(f"  Reordered {len(final_order)} columns")
    return df


# =============================================================================
# SAVE OUTPUT
# =============================================================================

def save_processed_data(df):
    """Save processed data as RTT.xlsx in data/processed/."""
    output_dir = get_project_root() / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "RTT.xlsx"
    df.to_excel(output_path, index=False, engine='openpyxl')
    print(f"\nSaved processed data to: {output_path}")
    print(f"Final shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return output_path


# =============================================================================
# PDF DATA DICTIONARY
# =============================================================================

def create_temporal_timeline():
    """
    Create a horizontal timeline diagram showing temporal ordering of variables.
    Returns a reportlab Drawing flowable (approx 480 x 240 pts).
    """
    W = 480
    H = 240
    d = Drawing(W, H)

    phases = [
        {
            'title': 'BEFORE DAY t',
            'subtitle': '(Yesterday Data)',
            'color': colors.HexColor('#2196F3'),
            'bg': colors.HexColor('#E3F2FD'),
            'vars': [
                'ACWR (x4)',
                'GPS % (x5)',
                'Training Intensity',
                'RPE Yesterday',
                'Activity Type Yest.',
                'Comment Yesterday',
                'Med. Avail / Attend.',
                'Raw GPS/HR Yest.',
                'Match Perf. Yest.',
            ],
        },
        {
            'title': 'MORNING (t)',
            'subtitle': '(Wellness Assessment)',
            'color': colors.HexColor('#FF9800'),
            'bg': colors.HexColor('#FFF3E0'),
            'vars': [
                'Wellness z-scores (x6)',
                'Status / Decrease',
                'Physical State',
                'Mental State',
                'Overall Wellbeing',
                'Days Since Game',
                'Days Until Match',
                'Match Day',
            ],
        },
        {
            'title': 'POST-ASSESSMENT',
            'subtitle': '(Treatment Decision)',
            'color': colors.HexColor('#E53935'),
            'bg': colors.HexColor('#FFEBEE'),
            'vars': [
                'Activity Type Today',
                'Selected',
                '',
                '(Coach decides',
                'session intensity',
                'after seeing',
                'morning data)',
            ],
        },
        {
            'title': "TODAY'S SESSION",
            'subtitle': '(Activity Execution)',
            'color': colors.HexColor('#757575'),
            'bg': colors.HexColor('#F5F5F5'),
            'vars': [
                'Training / Match / Rest',
                '',
                '(GPS & load data',
                'from this session',
                "becomes tomorrow's",
                "'yesterday' columns)",
            ],
        },
    ]

    n_phases = len(phases)
    box_w = 105
    arrow_gap = 20
    total_w = n_phases * box_w + (n_phases - 1) * arrow_gap
    x_start = (W - total_w) / 2
    box_h = 170
    y_base = 20

    for i, phase in enumerate(phases):
        x = x_start + i * (box_w + arrow_gap)

        d.add(Rect(x, y_base, box_w, box_h,
                   fillColor=phase['bg'], strokeColor=phase['color'],
                   strokeWidth=1.5, rx=5, ry=5))

        d.add(String(x + box_w / 2, y_base + box_h - 15,
                     phase['title'],
                     fontSize=7, fontName='Helvetica-Bold',
                     textAnchor='middle', fillColor=phase['color']))

        d.add(String(x + box_w / 2, y_base + box_h - 26,
                     phase['subtitle'],
                     fontSize=6, textAnchor='middle',
                     fillColor=colors.HexColor('#666666')))

        d.add(Line(x + 6, y_base + box_h - 31, x + box_w - 6,
                   y_base + box_h - 31,
                   strokeColor=phase['color'], strokeWidth=0.5))

        for j, var_name in enumerate(phase['vars']):
            if var_name:
                d.add(String(x + 7, y_base + box_h - 44 - j * 12,
                             var_name, fontSize=6,
                             fillColor=colors.HexColor('#333333')))

        if i < n_phases - 1:
            ax = x + box_w + 2
            ay = y_base + box_h / 2
            d.add(Line(ax, ay, ax + arrow_gap - 5, ay,
                       strokeColor=colors.HexColor('#999999'),
                       strokeWidth=1.2))
            d.add(Polygon(
                points=[ax + arrow_gap - 5, ay,
                        ax + arrow_gap - 9, ay + 3.5,
                        ax + arrow_gap - 9, ay - 3.5],
                fillColor=colors.HexColor('#999999'),
                strokeColor=colors.HexColor('#999999')))

    # "Night / Sleep" label on the first arrow
    night_x = x_start + box_w + arrow_gap / 2
    night_y = y_base + box_h / 2 + 22
    d.add(String(night_x, night_y, 'Night /',
                 fontSize=5.5, fontName='Helvetica-Oblique',
                 textAnchor='middle',
                 fillColor=colors.HexColor('#5C6BC0')))
    d.add(String(night_x, night_y - 8, 'Sleep',
                 fontSize=5.5, fontName='Helvetica-Oblique',
                 textAnchor='middle',
                 fillColor=colors.HexColor('#5C6BC0')))

    d.add(String(W / 2, y_base + box_h + 22,
                 'Temporal Timeline: Variables Within Each Row (Player-Day)',
                 fontSize=10, fontName='Helvetica-Bold',
                 textAnchor='middle',
                 fillColor=colors.HexColor('#1a1a2e')))

    return d


def generate_documentation_pdf(df, player_mapping, output_dir):
    """
    Auto-generate a color-coded PDF data dictionary for the processed dataset.

    Color coding by temporal group:
    - Green:  Identifiers
    - Blue:   Yesterday / historical (t-1) from Readiness_Data
    - Teal:   Yesterday from Raw_Data (GPS/HR)
    - Purple: Yesterday from Games (match performance)
    - Orange: Current (t) morning assessment
    - Red:    Post-assessment (t) data

    Underlined variables are engineered features not in the raw data.
    """
    print("\nGenerating PDF documentation...")
    pdf_path = Path(output_dir) / "RTT Data Dictionary.pdf"

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch
    )

    elements = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'CustomTitle', parent=styles['Heading1'],
        fontSize=18, textColor=colors.HexColor('#1a1a2e'),
        spaceAfter=10, fontName='Helvetica-Bold', alignment=TA_CENTER
    )
    overview_style = ParagraphStyle(
        'Overview', parent=styles['Normal'],
        fontSize=9, leading=12, textColor=colors.HexColor('#333333'),
        spaceAfter=12, alignment=TA_CENTER
    )
    cell_style = ParagraphStyle(
        'CellText', parent=styles['Normal'],
        fontSize=8, leading=10, textColor=colors.black
    )
    var_style_white = ParagraphStyle(
        'VarNameWhite', parent=styles['Normal'],
        fontSize=8, leading=10, fontName='Helvetica-Bold',
        textColor=colors.white
    )

    elements.append(Paragraph("RTT Data Dictionary — Processed Dataset", title_style))
    elements.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d')} &nbsp;|&nbsp; "
        f"Rows: {df.shape[0]:,} &nbsp;|&nbsp; Columns: {df.shape[1]} &nbsp;|&nbsp; "
        f"Players: {df['Player ID'].nunique()}",
        overview_style
    ))
    elements.append(Paragraph(
        "Each row = one player-day. Variables grouped by temporal position. "
        "Combines Readiness_Data, Raw_Data, Sessions, and Games.",
        overview_style
    ))

    # Legend
    legend_data = [[
        Paragraph('<font color="#4CAF50"><b>||</b></font> Identifiers', styles['Normal']),
        Paragraph('<font color="#2196F3"><b>||</b></font> Yesterday (RD)', styles['Normal']),
        Paragraph('<font color="#009688"><b>||</b></font> Yesterday (Raw)', styles['Normal']),
    ], [
        Paragraph('<font color="#7B1FA2"><b>||</b></font> Yesterday (Games)', styles['Normal']),
        Paragraph('<font color="#FF9800"><b>||</b></font> Morning (t)', styles['Normal']),
        Paragraph('<font color="#E53935"><b>||</b></font> Post-assessment', styles['Normal']),
    ]]
    legend_table = Table(legend_data, colWidths=[2.2 * inch] * 3)
    legend_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elements.append(legend_table)
    elements.append(Spacer(1, 0.15 * inch))

    elements.append(create_temporal_timeline())
    elements.append(Spacer(1, 0.15 * inch))

    # Colors
    c_ident      = colors.HexColor('#4CAF50')
    c_prev_rd1   = colors.HexColor('#2196F3')
    c_prev_raw   = colors.HexColor('#009688')
    c_prev_games = colors.HexColor('#7B1FA2')
    c_current    = colors.HexColor('#FF9800')
    c_post       = colors.HexColor('#E53935')

    # Variable definitions: [name, description, is_engineered, color]
    variables = [
        ['Date', 'Date of observation', False, c_ident],
        ['Playerkey', 'Anonymized player identifier (hash)', False, c_ident],
        ['Player ID', 'Sequential player ID (1-28)', True, c_ident],
        ['Position', 'Playing position (CD, ST, CDM, CAM, FB, WG, WB)', False, c_ident],
        ['Medical Availability\nLast 14 Days', 'Medical availability (%) over the last 14 days', False, c_prev_rd1],
        ['Club Attendance\nLast 14 Days', 'Club attendance (%) over the last 14 days', False, c_prev_rd1],
        ['Total Distance\n(ACWR) Yesterday', 'Total Distance ACWR (7:42 day EMA ratio)', False, c_prev_rd1],
        ['High Speed Distance\n(ACWR) Yesterday', 'High-Speed Distance (>19.8 km/h) ACWR', False, c_prev_rd1],
        ['High Decelerations\n(ACWR) Yesterday', 'High decelerations (>3 m/s\u00b2) ACWR', False, c_prev_rd1],
        ['Sprints (ACWR)\nYesterday', 'Sprint count (>25.2 km/h) ACWR', False, c_prev_rd1],
        ['Any ACWR Danger', 'Binary: 1 if any ACWR > 1.5 (Gabbett danger zone)', True, c_prev_rd1],
        ['Activity Type\nYesterday', 'Activity type from previous day (Training, Game, Rehab, etc.)', False, c_prev_rd1],
        ['Comment Yesterday', 'Staff notes on previous day status', False, c_prev_rd1],
        ['Comment Category\nYesterday', 'Categorized comment (recovery, discomfort, stiffness, etc.)', True, c_prev_rd1],
        ['Total Distance %\nYesterday', 'Total distance as % of personal match benchmark (capped at 250%)', False, c_prev_rd1],
        ['High Speed Distance\n% Yesterday', 'High-Speed Distance as % of personal benchmark (capped at 250%)', False, c_prev_rd1],
        ['High Decelerations\n% Yesterday', 'Decelerations as % of personal benchmark (capped at 250%)', False, c_prev_rd1],
        ['Sprints % Yesterday', 'Sprints as % of personal benchmark (capped at 250%)', False, c_prev_rd1],
        ['Max Velocity %\nYesterday', 'Max velocity as % of personal best (capped at 250%)', False, c_prev_rd1],
        ['Training Intensity\nYesterday', 'Composite: tanh(mean(TD%, HSD%, Dec%, Sprints%) / 100). Smooth soft cap in [0, 1): 100% match load -> 0.76, 130% -> 0.86. Excludes Max Velocity %.', True, c_prev_rd1],
        ['Perceived Exertion\nYesterday', 'RPE z-score (28-day rolling window)', False, c_prev_rd1],
        ['Total Minutes\nYesterday', 'Total session duration in minutes (summed across sessions). From Raw_Data, shifted +1 day.', False, c_prev_raw],
        ['Total Distance (m)\nYesterday', 'Total distance covered in metres (summed). From Raw_Data, shifted +1 day.', False, c_prev_raw],
        ['High Speed Distance (m)\nYesterday', 'High-speed distance in metres (summed). From Raw_Data, shifted +1 day.', False, c_prev_raw],
        ['Avg Heart Rate\nYesterday', 'Average heart rate (bpm, weighted mean by session duration). From Raw_Data, shifted +1 day.', False, c_prev_raw],
        ['Heart Rate Exertion\nYesterday', 'Heart rate exertion score (weighted mean). From Raw_Data, shifted +1 day.', False, c_prev_raw],
        ['Match HID Per BIP\nYesterday', 'High-intensity distance per Ball-In-Play minute (m). Only filled day after match. From Games, shifted +1 day.', False, c_prev_games],
        ['Match HIE Per BIP\nYesterday', 'High-intensity efforts per Ball-In-Play minute. Only filled day after match. From Games, shifted +1 day.', False, c_prev_games],
        ['Match Minutes Played\nYesterday', 'Total minutes played in match. Only filled day after match. From Games, shifted +1 day.', False, c_prev_games],
        ['Match Intensity\nYesterday', 'Causal outcome Y: geometric mean of HID Per BIP and HIE Per BIP, multiplied by sqrt(clip(minutes_played, 15, 90)/90): playing time clamped to [15, 90] min. Continuous, range ~[0, ∞). Only filled day after match.', True, c_prev_games],
        ['Status', 'Current medical status (Available, Attention, Injured, Sick, Absent)', False, c_current],
        ['Status Decrease', 'Binary: 1 if status worsened vs previous day', True, c_current],
        ['Fatigue (z)', 'Self-reported fatigue z-score (28-day rolling window)', False, c_current],
        ['Readiness (z)', 'Self-reported readiness to train z-score', False, c_current],
        ['Soreness (z)', 'Self-reported muscle soreness z-score', False, c_current],
        ['Physical State', 'Composite: mean of Fatigue, Readiness, Soreness z-scores', True, c_current],
        ['Sleep Quality (z)', 'Self-reported sleep quality z-score', False, c_current],
        ['Stress (z)', 'Self-reported stress level z-score', False, c_current],
        ['Mood (z)', 'Self-reported mood z-score', False, c_current],
        ['Mental State', 'Composite: mean of Sleep Quality, Stress, Mood z-scores', True, c_current],
        ['Overall Wellbeing', 'Composite: mean of all 6 wellness z-scores', True, c_current],
        ['Days Since Game', 'Days since last completed match (min=1, never 0)', True, c_current],
        ['Days Until Match', 'Days until next scheduled match (0 on match day for selected players)', True, c_current],
        ['Match Day', 'Team-level: 1 if scheduled match today (known from fixture list in advance)', True, c_current],
        ['Activity Type Today', 'Session type on day t, assigned after morning assessment. Derived from next row.', True, c_post],
        ['Selected', 'Player-level: 1 if selected for match, 0 if not, NaN on non-match days', True, c_post],
    ]

    def build_var_row(name, desc, is_engineered):
        if is_engineered:
            name_para = Paragraph(f'<u>{name}</u>', var_style_white)
        else:
            name_para = Paragraph(name, var_style_white)
        desc_para = Paragraph(desc, cell_style)
        return [name_para, desc_para]

    table_data = [[
        Paragraph('<b>Variable</b>', ParagraphStyle('H', fontSize=10, textColor=colors.white)),
        Paragraph('<b>Description</b>', ParagraphStyle('H', fontSize=10, textColor=colors.white))
    ]]
    row_colors = []
    for var in variables:
        table_data.append(build_var_row(var[0], var[1], var[2]))
        row_colors.append(var[3])

    var_table = Table(table_data, colWidths=[1.8 * inch, 4.8 * inch])
    style_cmds = [
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2d2d2d')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, 0), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('TOPPADDING', (0, 0), (-1, 0), 10),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 1), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
        ('BACKGROUND', (1, 1), (1, -1), colors.HexColor('#fafafa')),
    ]
    for i, color in enumerate(row_colors, start=1):
        style_cmds.append(('BACKGROUND', (0, i), (0, i), color))
    var_table.setStyle(TableStyle(style_cmds))
    elements.append(var_table)

    elements.append(Spacer(1, 0.15 * inch))
    note_style = ParagraphStyle(
        'Note', parent=styles['Normal'], fontSize=9,
        textColor=colors.HexColor('#555555'), alignment=TA_LEFT
    )
    elements.append(Paragraph(
        "<b>Note:</b> <u>Underlined variables</u> are engineered features "
        "created during preprocessing (not present in the raw data).",
        note_style
    ))

    elements.append(Spacer(1, 0.2 * inch))
    src_title = ParagraphStyle(
        'SrcTitle', parent=styles['Heading2'], fontSize=12,
        textColor=colors.HexColor('#1a1a2e'), spaceBefore=6,
        spaceAfter=8, fontName='Helvetica-Bold'
    )
    elements.append(Paragraph("Data Sources", src_title))

    src_style = ParagraphStyle(
        'SrcText', parent=styles['Normal'], fontSize=9,
        leading=12, spaceBefore=3, spaceAfter=3,
        textColor=colors.HexColor('#333333')
    )
    sources = [
        ("<b>Readiness_Data.xlsx</b> — Base dataset. Daily player-day observations "
         "with ACWR, wellness z-scores, GPS %, RPE, medical status, and activity type."),
        ("<b>Raw_Data.xlsx</b> — Session-level GPS and heart rate data. "
         "Filtered to base dataset players, aggregated per player-day, shifted +1 day "
         "to become 'yesterday' data."),
        ("<b>Sessions.xlsx</b> — Team-level session metadata. "
         "Loaded for reference; match day identification from Activity Type Today."),
        ("<b>Games.xlsx</b> — Match performance data. "
         "Filtered to base dataset players, shifted +1 day to become 'yesterday' match data."),
    ]
    for s in sources:
        elements.append(Paragraph(s, src_style))

    elements.append(Spacer(1, 0.2 * inch))
    elements.append(Paragraph("Definitions", src_title))

    def_style = ParagraphStyle(
        'Definition', parent=styles['Normal'], fontSize=9,
        leading=12, spaceBefore=4, spaceAfter=4,
        textColor=colors.HexColor('#333333')
    )
    definitions = [
        ("<b>ACWR (Acute:Chronic Workload Ratio)</b> — Compares short-term (acute, 7-day) "
         "to long-term (chronic, 42-day) training load using EMA. Sweet spot: 0.8-1.3. "
         "Ratios >1.5 indicate elevated injury risk."),
        ("<b>EMA (Exponential Moving Average)</b> — Weights recent observations more heavily "
         "than older ones, providing a responsive measure of current fitness and fatigue states."),
        ("<b>Personal Benchmark</b> — Average of the player's five best match performances. "
         "Percentage metrics are relative to this individual benchmark."),
        ("<b>Z-scores</b> — Standardized scores where 0 = player's individual mean and "
         "+/-1 = one standard deviation. Based on 28-day rolling window per player."),
        ("<b>Status Decrease</b> — Binary indicator (0/1) flagging when a player's medical "
         "status worsened vs the previous day."),
        ("<b>Days Since Game</b> — Days since the player's last completed match. "
         "Measured in the morning. Minimum = 1, never 0."),
        ("<b>Days Until Match</b> — Days until the next scheduled match. "
         "Equals 0 on match day. Captures the weekly microcycle phase."),
        ("<b>BIP (Ball In Play)</b> — Time during a match when the ball is actively in play, "
         "used to normalize physical intensity metrics for fair comparison across matches."),
    ]
    for defn in definitions:
        elements.append(Paragraph(defn, def_style))

    # Player ID mapping
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(Paragraph("Player ID Mapping", src_title))

    mapping_data = [[
        Paragraph('<b>Player ID</b>', ParagraphStyle('H', fontSize=9, textColor=colors.white)),
        Paragraph('<b>Playerkey (hash)</b>', ParagraphStyle('H', fontSize=9, textColor=colors.white))
    ]]
    for pk, pid in sorted(player_mapping.items(), key=lambda x: x[1]):
        mapping_data.append([
            Paragraph(str(pid), cell_style),
            Paragraph(str(pk), cell_style)
        ])

    map_table = Table(mapping_data, colWidths=[1.0 * inch, 5.6 * inch])
    map_style = [
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2d2d2d')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]
    for i in range(1, len(mapping_data)):
        bg = colors.HexColor('#f5f5f5') if i % 2 == 0 else colors.white
        map_style.append(('BACKGROUND', (0, i), (-1, i), bg))
    map_table.setStyle(TableStyle(map_style))
    elements.append(map_table)

    doc.build(elements)
    print(f"  PDF documentation saved to: {pdf_path}")
    return pdf_path


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def preprocess_data(date_min='2025-07-27', date_max='2026-02-28', benchmark_cap=250):
    """
    Main preprocessing pipeline — orchestrates all transformation steps.

    Parameters
    ----------
    date_min : str or None, default '2025-07-27'
        Start date filter (inclusive). Set to None to keep all early rows.
    date_max : str or None, default '2026-02-28'
        End date filter (inclusive). Set to None to keep all late rows.
    benchmark_cap : int or None, default 250
        Cap for GPS benchmark % columns. Set to None to skip capping.

    Steps
    -----
    1.  Load all raw xlsx files
    2.  Basic transforms on Readiness_Data (player IDs, renaming, %)
    3.  Comment categorization
    4.  Merge Raw_Data GPS/HR columns (shifted +1 day)
    5.  Merge Games match performance columns (shifted +1 day)
    6.  Composite wellness scores
    7.  Temporal features (Activity Type Today, Days Since/Until Game)
    8.  Match day features
    9.  Status Decrease
    10. ACWR danger flag
    11. Training Intensity Yesterday composite
    12. Benchmark % capping
    13. Date range filter
    14. Column reordering
    15. Save RTT.xlsx + PDF

    Returns
    -------
    pd.DataFrame
        The fully preprocessed DataFrame.
    """
    print("=" * 80)
    print("OHL PLAYER READINESS - MULTI-DATASET PREPROCESSING PIPELINE")
    print("=" * 80)
    if date_min or date_max:
        print(f"  Date range: {date_min} \u2192 {date_max}")
    if benchmark_cap is not None:
        print(f"  Benchmark % cap: {benchmark_cap}%")

    # 1. Load all data
    data = load_all_raw_data()
    df = data['rd1'].copy()

    # 2. Basic transformations
    df, player_mapping = map_player_ids(df)
    df = rename_columns(df)
    df = clean_percentage_columns(df)

    # 3. Comment analysis
    df = add_comment_category(df)

    # 4. Merge Raw_Data (GPS/HR — shifted to "yesterday")
    df = merge_raw_data(df, data['raw'], player_mapping)

    # 5. Merge Games (match performance — shifted to "yesterday")
    df = merge_games_data(df, data['games'])

    # 6. Composite scores
    df = add_physical_state(df)
    df = add_mental_state(df)
    df = add_overall_wellbeing(df)

    # 7. Temporal features
    df = add_activity_type_today(df)
    df = add_days_since_game(df)
    df = add_days_until_match(df)

    # 8. Match day features (depends on Activity Type Today)
    df = add_match_day_and_selected(df)

    # 9. Status features
    df = add_status_decrease(df)

    # 10. ACWR danger flag
    df = add_any_acwr_danger(df)

    # 11. Training intensity composite
    df = add_training_intensity_yesterday(df)

    # 12. Benchmark % capping
    if benchmark_cap is not None:
        df = cap_benchmark_percentages(df, cap=benchmark_cap)

    # 13. Date range filter
    df = filter_date_range(df, date_min=date_min, date_max=date_max)

    # 14. Final column ordering
    df = reorder_columns(df)

    # 15. Save results
    output_path = save_processed_data(df)
    generate_documentation_pdf(df, player_mapping, output_path.parent)

    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE")
    print(f"  Total features: {df.shape[1]}")
    print(f"  Total observations: {df.shape[0]:,}")
    print(f"  Date range: {df['Date'].min().date()} \u2192 {df['Date'].max().date()}")
    print(f"  Output: {output_path}")
    print("=" * 80)

    return df


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    preprocess_data()
