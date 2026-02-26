"""
Data Preprocessing Pipeline
Cleans and transforms raw readiness data for analysis.

This module takes the raw CSV exported from the OHL player monitoring system and applies a series of
transformations to produce a clean, feature-engineered dataset ready for the data loader.

Pipeline Summary (executed in order by preprocess_data()):
    1. Load raw CSV
    2. Map complex Playerkey strings → simple sequential Player IDs (1, 2, 3, ...)
    3. Rename cryptic column abbreviations → human-readable names
    4. Clean percentage columns (string → integer conversion)
    5. Categorize free-text comments into keyword-based categories
    6. Create composite wellness scores (Physical State, Mental State, Overall Wellbeing)
    7. Create temporal features (Activity Type Today, Days Since Game, Days Until Match)
    8. Create match day features (Match Day team-level flag, Selected player-level flag)
    9. Detect status decrease transitions
    10. Flag dangerous ACWR values (>1.5 threshold from Gabbett 2016)
    11. Reorder columns into temporal groups
    12. Save processed CSV + auto-generate PDF data dictionary

=== TEMPORAL SEMANTICS ===

Each row represents one player-day (date t). Within that row, variables have
different temporal positions:

    BEFORE day t:   Activity Type Yesterday, GPS %, ACWR, Comments, RPE
                    (these describe what happened on day t-1)

    MORNING of t:   Wellness z-scores, Status, Days Since Game, Days Until Match
                    (measured before any training decisions are made)

    AFTER morning:  Activity Type Today
                    (the session type assigned after the morning assessment;
                    this is derived from the NEXT row's Activity Type Yesterday,
                    so it represents data from one timestep ahead — t+1 relative
                    to the assessment data in the same row)

Understanding this temporal ordering is important for correctly specifying
which variables are available at which point in time.

Input:  data/raw/Readiness_Data.csv
Output: data/processed/Readiness_Data.csv
        data/processed/Preprocessing_Documentation.pdf

Usage:
    # As a standalone script:
    python src/data/data_preprocessing.py

    # As an importable module:
    from src.data.data_preprocessing import preprocess_data
    df = preprocess_data()
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, 
    PageBreak, KeepTogether
)
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER


# =============================================================================
# DATA LOADING
# =============================================================================

def load_raw_data():
    """
    Load raw readiness data from the CSV file.

    Uses pathlib for cross-platform path resolution. The path is computed
    relative to this script's location (src/data/), going up two levels
    to the project root, then into data/raw/.

    The 'Date' column is parsed as datetime during loading to enable
    temporal operations (sorting, date arithmetic) downstream.

    Returns
    -------
    pd.DataFrame
        Raw data with Date column as datetime dtype.
    """
    # Navigate from src/data/ → src/ → project_root/
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    input_path = project_root / "data" / "raw" / "Readiness_Data.csv"

    print(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path, parse_dates=['Date'])
    print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

    return df


# =============================================================================
# BASIC TRANSFORMATIONS
# =============================================================================

def map_player_ids(df):
    """
    Convert complex player keys to simple sequential IDs (1, 2, 3, ...).

    The raw data uses anonymized 'Playerkey' strings that are long and
    hard to work with. This function creates a new 'Player ID' column
    with simple integer IDs while preserving the original Playerkey
    for traceability.

    The mapping is deterministic: players are sorted alphabetically by
    their Playerkey, then assigned IDs starting from 1. This ensures
    the same player always gets the same ID across runs.

    Parameters
    ----------
    df : pd.DataFrame
        Raw data with 'Playerkey' column.

    Returns
    -------
    tuple of (pd.DataFrame, dict)
        - DataFrame with new 'Player ID' column added
        - Dictionary mapping original Playerkey → new Player ID
    """
    print("\nMapping player IDs...")

    # Sort alphabetically so the mapping is deterministic across runs
    unique_players = sorted(df['Playerkey'].unique())
    # Create mapping: original_key → sequential integer (starting at 1)
    player_mapping = {old_key: idx + 1 for idx, old_key in enumerate(unique_players)}

    # Apply mapping to create the new column
    df['Player ID'] = df['Playerkey'].map(player_mapping)

    print(f"  Mapped {len(unique_players)} players to IDs 1-{len(unique_players)}")

    return df, player_mapping


def rename_columns(df):
    """
    Rename cryptic column abbreviations to human-readable names.

    The raw data uses shorthand column names from the monitoring system
    (e.g., 'TD' for Total Distance, 'HSD' for High Speed Distance).
    This function expands them into descriptive names that include:
    - What the metric measures
    - The calculation method (e.g., ACWR = Acute:Chronic Workload Ratio)
    - The temporal reference (e.g., 'Yesterday' = data from previous day)

    Note: The 'Yesterday' suffix indicates that these metrics were recorded
    on the day BEFORE the current observation row. This is important for
    understanding the temporal structure of the data — the model uses
    yesterday's workload/activity data to predict today's status.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with original abbreviated column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with renamed columns.
    """
    print("\nRenaming columns...")

    rename_dict = {
        'MA%': 'Medical Availability Last 14 Days',       # Rolling 14-day medical availability %
        'Att%': 'Club Attendance Last 14 Days',            # Rolling 14-day attendance %
        'POS': 'Position',                                 # Playing position (CD, ST, CAM, etc.)
        'TD': 'Total Distance (ACWR) Yesterday',           # Total Distance ACWR ratio
        'HSD': 'High Speed Distance (ACWR) Yesterday',     # High Speed Distance (>19.8 km/h) ACWR
        'Dec >3ms²': 'High Decelerations (ACWR) Yesterday',  # Deceleration events (>3 m/s²) ACWR
        'Sprints': 'Sprints (ACWR) Yesterday',             # Sprint count (>25.2 km/h) ACWR
        'Reason': 'Activity Type Yesterday',               # What the player did yesterday
        'Comment': 'Comment Yesterday',                    # Free-text staff notes from yesterday
        'TD%': 'Total Distance % Yesterday',               # Total distance as % of personal benchmark
        'HSD%': 'High Speed Distance % Yesterday',         # HSD as % of personal benchmark
        'Dec >3ms²%': 'High Decelerations % Yesterday',    # Decelerations as % of personal benchmark
        'Sprints%': 'Sprints % Yesterday',                 # Sprints as % of personal benchmark
        'Max, Velocity%': 'Max Velocity % Yesterday',      # Max velocity as % of personal best
        'rpe (z)': 'Perceived Exertion Yesterday'          # RPE z-score (based on 28-day rolling window)
    }

    df = df.rename(columns=rename_dict)
    print(f"  Renamed {len(rename_dict)} columns")

    return df


def clean_percentage_columns(df):
    """
    Convert percentage columns from strings to integers.

    The raw data sometimes stores percentage values as strings (e.g., '85.0')
    instead of numbers. This function safely converts them to integers by:
    1. Converting string → float (handles decimal representations)
    2. Converting float → int (e.g., 85.0 → 85)
    3. Preserving NaN/empty values as-is

    This is applied to the rolling 14-day medical availability and attendance
    columns, which represent percentages (0-100).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with percentage columns as potential strings.

    Returns
    -------
    pd.DataFrame
        DataFrame with cleaned integer percentage columns.
    """
    print("\nCleaning percentage columns...")

    percentage_cols = ['Medical Availability Last 14 Days', 'Club Attendance Last 14 Days']

    for col in percentage_cols:
        if col in df.columns:
            # Two-step conversion: str → float → int (handles '85.0' → 85)
            # Lambda preserves NaN and empty strings without crashing
            df[col] = df[col].apply(lambda x: int(float(x)) if pd.notna(x) and x != '' else x)
            print(f"  {col}: converted to integer type")

    print(f"  Cleaned {len([c for c in percentage_cols if c in df.columns])} percentage columns")

    return df


# =============================================================================
# COMMENT CATEGORIZATION
# =============================================================================

def categorize_comment(comment):
    """
    Categorize a single free-text comment into a predefined category.

    Staff members write free-text comments about player status. This function
    converts those unstructured notes into categorical labels by searching
    for specific keywords. This makes the information usable as a feature
    in ML models (which can't process free text directly without NLP).

    Categorization rules:
    - 'recovery'   : comment mentions recovery
    - 'discomfort'  : comment mentions discomfort
    - 'stiffness'   : comment mentions stiffness
    - 'sick'        : comment mentions viral, infection, symptoms, or cold
    - 'hydrops'     : comment mentions hydrops (joint swelling)
    - 'cramp'       : comment mentions cramp
    - 'empty'       : no comment was written (NaN or empty string)
    - 'other'       : comment exists but doesn't match any keyword,
                      OR matches multiple conflicting categories

    The multi-match → 'other' rule avoids ambiguity when a comment
    contains keywords from multiple categories (e.g., "stiffness and cramp").

    Parameters
    ----------
    comment : str or NaN
        A single free-text comment string.

    Returns
    -------
    str
        One of: 'recovery', 'discomfort', 'stiffness', 'sick', 'hydrops',
        'cramp', 'empty', or 'other'.
    """
    # Handle missing or empty comments
    if pd.isna(comment) or comment == '':
        return 'empty'

    comment_lower = str(comment).lower()

    # Define keyword-to-category mapping
    categories = {
        'recovery': ['recovery'],
        'discomfort': ['discomfort'],
        'stiffness': ['stiffness'],
        'sick': ['viral', 'infection', 'symptoms', 'cold'],
        'hydrops': ['hydrops'],
        'cramp': ['cramp']
    }

    # Check which categories match the comment
    matches = []
    for category, keywords in categories.items():
        if any(keyword in comment_lower for keyword in keywords):
            matches.append(category)

    # Return based on number of matches:
    # 0 matches → 'other' (comment exists but doesn't fit any category)
    # 1 match   → that category (unambiguous)
    # 2+ matches → 'other' (ambiguous — avoids assigning wrong label)
    if len(matches) == 0:
        return 'other'
    elif len(matches) == 1:
        return matches[0]
    else:
        return 'other'


def add_comment_category(df):
    """
    Add 'Comment Category Yesterday' column by categorizing free-text comments.

    Applies the categorize_comment() function to every row's comment text,
    producing a new categorical column suitable for ML encoding (label or one-hot).

    This is an ENGINEERED FEATURE — not present in the raw data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'Comment Yesterday' column containing free-text.

    Returns
    -------
    pd.DataFrame
        DataFrame with new 'Comment Category Yesterday' column.
    """
    print("\nAdding Comment Category Yesterday...")

    # Apply keyword categorization to each comment
    df['Comment Category Yesterday'] = df['Comment Yesterday'].apply(categorize_comment)

    # Print distribution for transparency
    category_counts = df['Comment Category Yesterday'].value_counts()
    print(f"  Added Comment Category Yesterday:")
    for category, count in category_counts.items():
        print(f"    - {category}: {count}")

    return df


# =============================================================================
# COMPOSITE SCORES
# =============================================================================

def add_physical_state(df):
    """
    Add 'Physical State' composite score.

    Computed as the mean of the three physical wellness z-scores:
    Fatigue, Readiness, and Soreness. These z-scores are normalized
    per player over a 28-day rolling window (done upstream in the
    monitoring system), so this composite represents a player's
    overall physical state relative to their own recent baseline.

    Note: pandas .mean(axis=1) automatically handles NaN by computing
    the mean over available (non-NaN) values in each row. If all 3
    values are NaN, the result is NaN.

    This is an ENGINEERED FEATURE — not present in the raw data.
    """
    print("\nAdding Physical State...")

    physical_cols = ['Fatigue (z)', 'Readiness (z)', 'Soreness (z)']
    # Row-wise mean across the 3 physical z-scores
    df['Physical State'] = df[physical_cols].mean(axis=1)

    n_valid = df['Physical State'].notna().sum()
    print(f"  Added Physical State ({n_valid:,} valid entries)")

    return df


def add_mental_state(df):
    """
    Add 'Mental State' composite score.

    Computed as the mean of the three psychological wellness z-scores:
    Sleep Quality, Stress, and Mood. This provides a single indicator
    of a player's mental/psychological readiness.

    This is an ENGINEERED FEATURE — not present in the raw data.
    """
    print("\nAdding Mental State...")

    mental_cols = ['Sleep Quality (z)', 'Stress (z)', 'Mood (z)']
    # Row-wise mean across the 3 mental/psychological z-scores
    df['Mental State'] = df[mental_cols].mean(axis=1)

    n_valid = df['Mental State'].notna().sum()
    print(f"  Added Mental State ({n_valid:,} valid entries)")

    return df


def add_overall_wellbeing(df):
    """
    Add 'Overall Wellbeing' composite score.

    Computed as the mean of ALL SIX wellness z-scores (3 physical + 3 mental).
    This is the most aggregated wellness indicator, capturing both physical
    and psychological state in a single number.

    Interpretation: values near 0 = average for that player; positive = above
    average; negative = below average (all relative to the player's own
    28-day rolling baseline).

    This is an ENGINEERED FEATURE — not present in the raw data.
    """
    print("\nAdding Overall Wellbeing...")

    wellbeing_cols = [
        'Fatigue (z)', 'Readiness (z)', 'Soreness (z)',       # Physical
        'Sleep Quality (z)', 'Stress (z)', 'Mood (z)'         # Mental
    ]
    # Row-wise mean across all 6 z-scores
    df['Overall Wellbeing'] = df[wellbeing_cols].mean(axis=1)

    n_valid = df['Overall Wellbeing'].notna().sum()
    print(f"  Added Overall Wellbeing ({n_valid:,} valid entries)")

    return df


# =============================================================================
# TEMPORAL FEATURES
# =============================================================================

def add_activity_type_today(df):
    """
    Add 'Activity Type Today' — the session type assigned to the player on day t.

    === TEMPORAL SEMANTICS ===

    This variable captures the TYPE of activity/session the player performs on
    day t (e.g., Training, Game, Rehab, Day off). It is determined AFTER the
    morning wellness assessment has been completed.

    The temporal ordering within a single day (row at date t) is:

        1. Morning: Player fills wellness questionnaire → Fatigue(z), Readiness(z),
           Soreness(z), Sleep Quality(z), Stress(z), Mood(z), Status
        2. Morning: Staff records Days Since Game, Medical Availability, etc.
        3. THEN: Coach decides the session type → Activity Type Today
        4. Player performs the activity → GPS data recorded (appears as "Yesterday"
           data in TOMORROW's row at t+1)

    Because this variable is only known after the morning assessment, it
    represents data from a later point in time than all other variables in
    the same row. It is effectively a t+1 data point relative to the
    assessment variables.

    === HOW IT IS DERIVED ===

    The raw data records 'Activity Type Yesterday' for each row. To get
    today's activity type, we look at the NEXT row's 'Activity Type Yesterday'
    for the same player (since tomorrow's "yesterday" is today's "today").

    This is done using .shift(-1) within each player group:
    - shift(-1) pulls the value from the NEXT chronological row
    - groupby('Player ID') ensures we never leak data across players
    - The last row for each player gets NaN (no future data available)

    This is an ENGINEERED FEATURE — not present in the raw data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'Player ID', 'Date', and 'Activity Type Yesterday' columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with new 'Activity Type Today' column.
    """
    print("\nAdding Activity Type Today...")

    # Sort by player and date to ensure chronological order within each player
    df_sorted = df.sort_values(['Player ID', 'Date']).copy()

    # shift(-1): pull the NEXT row's 'Activity Type Yesterday' within each player group.
    # Since tomorrow's "yesterday" is today's "today", this gives Activity Type Today.
    df_sorted['Activity Type Today'] = df_sorted.groupby('Player ID')['Activity Type Yesterday'].shift(-1)

    # Date continuity guard: Activity Type Today is only valid if the very next row
    # for this player is exactly date+1. If there is a gap (the player was absent the
    # following day and no row exists for that day), the shift pulls the wrong activity.
    # Example: row at Friday (t) pulls Monday's Activity Type Yesterday as "Today",
    # which would incorrectly label Friday's treatment as Monday's session type.
    # Fix: NaN out any row where the next row is not strictly the next calendar day.
    df_sorted['_next_date'] = df_sorted.groupby('Player ID')['Date'].shift(-1)
    gap_mask = df_sorted['_next_date'] != (df_sorted['Date'] + pd.Timedelta(days=1))
    df_sorted.loc[gap_mask, 'Activity Type Today'] = np.nan
    df_sorted = df_sorted.drop(columns=['_next_date'])

    # Map the computed values back to the original DataFrame's row order
    # using a temporary composite key (Player_ID + Date) as the join key
    df_sorted['_temp_key'] = df_sorted['Player ID'].astype(str) + '_' + df_sorted['Date'].astype(str)
    df['_temp_key'] = df['Player ID'].astype(str) + '_' + df['Date'].astype(str)

    activity_mapping = df_sorted.set_index('_temp_key')['Activity Type Today'].to_dict()
    df['Activity Type Today'] = df['_temp_key'].map(activity_mapping)

    # Clean up temporary column
    df = df.drop(columns=['_temp_key'])

    n_valid = df['Activity Type Today'].notna().sum()
    print(f"  Added Activity Type Today ({n_valid:,} valid entries)")

    return df


def add_match_day_and_selected(df):
    """
    Add 'Match Day' (team-level) and 'Selected' (player-level) features.

    === MATCH DAY (team-level, binary) ===

    A date is a match day if ANY player in the squad has Activity Type Today
    equal to exactly 'Game' on that date. This is a TEAM-LEVEL variable:
    even if a specific player does not play, the day is still a match day
    because the team has a match.

    IMPORTANT: Only the exact activity type 'Game' counts as an OHL first-team
    match. Other activity types that contain the word 'game' are explicitly
    excluded:
    - 'Youth training or game' — youth academy, NOT an OHL match
    - 'National team training/game' — international duty, NOT an OHL match

    Values:
        0 = no match scheduled for the team on this date
        1 = the team has a match on this date

    === SELECTED (player-level) ===

    Indicates whether the individual player was selected to play in the match.
    This is only meaningful on match days.

    Values:
        NaN = no match on this date (Match Day = 0), so selection is not applicable
        1   = player was selected (their Activity Type Today is a game/match)
        0   = player was NOT selected (team has a match, but this player did
              something else: training, rehab, day off, etc.)

    === TEMPORAL POSITION ===

    Both variables are derived from Activity Type Today, which is a post-assessment
    variable (determined after the morning wellness assessment). Therefore Match Day
    and Selected are also post-assessment variables — they describe what happens
    AFTER the morning assessment on day t.

    Note: Activity Type Today may be NaN for some rows (last row per player).
    On those rows, Match Day could be 1 if other players have a game, but
    Selected will be NaN (we don't know what the player did).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'Date' and 'Activity Type Today' columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with new 'Match Day' and 'Selected' columns.
    """
    print("\nAdding Match Day and Selected...")

    # Step 1: Identify match days (team-level)
    # For each date, check if ANY player has a game/match as Activity Type Today
    game_mask = df['Activity Type Today'].str.strip().str.lower() == 'game'
    match_dates = set(df.loc[game_mask, 'Date'].unique())

    df['Match Day'] = df['Date'].apply(lambda d: 1 if d in match_dates else 0)

    # Step 2: Assign Selected (player-level)
    # NaN when no match, 1 if player has game, 0 if player does something else
    def assign_selected(row):
        if row['Match Day'] == 0:
            return np.nan
        # Match day: check if this player's activity is a game
        activity = row['Activity Type Today']
        if pd.isna(activity):
            return np.nan  # Unknown activity — can't determine selection
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
    Add 'Days Since Game' — number of calendar days since the player's LAST COMPLETED match.

    === TEMPORAL SEMANTICS (CRITICAL) ===

    This is a "morning of day t" variable. It represents how many days have
    elapsed since the player's most recent COMPLETED game, measured at the
    time the wellness questionnaire is filled in (morning of day t), BEFORE
    any activity/treatment happens on day t.

    This means: on a match day itself, the game has NOT yet been played when
    this variable is measured. Therefore Days Since Game refers to the
    PREVIOUS match, NOT today's upcoming match. Consequently, this value
    can NEVER be 0.

    === HOW IT WORKS ===

    Each row's 'Activity Type Yesterday' tells us what the player did on
    day t-1. If Activity Type Yesterday is exactly 'Game' (OHL first-team
    match only — excludes 'Youth training or game' and 'National team
    training/game'), the game was played on day t-1. Therefore:

    - The ACTUAL game date = row['Date'] - 1 day (for rows where yesterday was a game)
    - Days Since Game at row t = (row['Date'] - most_recent_game_BEFORE_today).days

    IMPORTANT: We only count games that happened STRICTLY BEFORE the current
    date (i.e., actual_game_date < current_date). This ensures:
    - On a match day: Days Since Game = days since the PREVIOUS match (never 0)
    - The day after a match: Days Since Game = 1

    Example for Player 5 (matches on Jan 10 and Jan 14):
        Date        Act Yesterday    Act Today    Actual Game    Days Since Game
        Jan 10      Training         Game         -              NaN (no prior game)
        Jan 11      Game             Training     Jan 10         1   ← game was yesterday
        Jan 12      Training         Training     Jan 10         2
        Jan 13      Training         Training     Jan 10         3
        Jan 14      Training         Game         Jan 10         4   ← match day! But game not yet played
        Jan 15      Game             Training     Jan 14         1   ← yesterday's game now counts

    This feature captures post-match recovery state at time of assessment:
    - Low values (1-2): recently played, likely fatigued
    - Medium values (3-5): standard recovery window
    - High values (7+): extended gap between matches

    NaN is assigned when no prior game exists for that player.

    This is an ENGINEERED FEATURE — not present in the raw data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'Player ID', 'Date', and 'Activity Type Yesterday' columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with new 'Days Since Game' column (float, NaN for no prior game).
        The minimum value is 1 (never 0).
    """
    print("\nAdding Days Since Game...")

    df_sorted = df.sort_values(['Player ID', 'Date']).copy()

    def calc_days_since_game(group):
        """
        Inner function applied per player group.

        For each row in the player's timeline:
        1. Find rows where Activity Type Yesterday was exactly 'Game'
           (OHL first-team match only — excludes youth/national team)
        2. Compute the ACTUAL game date = row_date - 1 day
        3. For each row, find the most recent actual game date STRICTLY BEFORE current date
        4. Compute days elapsed = current_date - actual_game_date
        """
        # Identify which rows report an OHL first-team game as yesterday's activity
        is_game_yesterday = group['Activity Type Yesterday'].str.strip().str.lower() == 'game'

        # The ACTUAL game date is the day before the row that records it
        # (because Activity Type Yesterday on date t means the activity on t-1)
        actual_game_dates = group.loc[is_game_yesterday, 'Date'] - pd.Timedelta(days=1)

        result = pd.Series(index=group.index, dtype=float)
        for idx, row in group.iterrows():
            current_date = row['Date']
            # STRICTLY BEFORE current date — game on same calendar day hasn't happened yet
            # because this variable is measured in the morning before any activity
            prior_games = actual_game_dates[actual_game_dates < current_date]
            if len(prior_games) > 0:
                days = (current_date - prior_games.max()).days
                result[idx] = days  # Minimum is 1 (game was yesterday)
            else:
                result[idx] = np.nan
        return result

    # Apply per-player to prevent cross-player contamination
    df_sorted['Days Since Game'] = df_sorted.groupby('Player ID', group_keys=False).apply(calc_days_since_game, include_groups=False)

    # Map back to original DataFrame order using _temp_key pattern
    df_sorted['_temp_key'] = df_sorted['Player ID'].astype(str) + '_' + df_sorted['Date'].astype(str)
    df['_temp_key'] = df['Player ID'].astype(str) + '_' + df['Date'].astype(str)
    days_mapping = df_sorted.set_index('_temp_key')['Days Since Game'].to_dict()
    df['Days Since Game'] = df['_temp_key'].map(days_mapping)
    df = df.drop(columns=['_temp_key'])

    n_valid = df['Days Since Game'].notna().sum()
    n_zeros = (df['Days Since Game'] == 0).sum()
    print(f"  Added Days Since Game ({n_valid:,} valid entries, {n_zeros} zeros [should be 0])")
    if n_zeros > 0:
        print(f"  WARNING: Found {n_zeros} zero values — this should not happen!")

    return df


def add_days_until_match(df):
    """
    Add 'Days Until Match' — number of calendar days until the player's NEXT game.

    === TEMPORAL SEMANTICS ===

    This is a "today" (time t) variable measured on the morning of day t,
    BEFORE the treatment (Activity Type Today) is applied. It represents
    how many days remain until the player's next scheduled match.

    This is derived from the Activity Type Today column: if Activity Type Today
    on some future date = 'Game', then on the current date we compute
    the number of days until that game.

    On a match day itself (when Activity Type Today = 'Game'), Days Until Match = 0.

    Example for Player 5 (games on Jan 10 and Jan 17):
        Date        Act Today    Days Until Match
        Jan 07      Training     3
        Jan 08      Training     2
        Jan 09      Training     1
        Jan 10      Game         0    ← match day
        Jan 11      Training     6
        Jan 12      Training     5
        ...
        Jan 17      Game         0    ← next match day

    If the player has no future matches in the dataset, Days Until Match is NaN.

    This feature is critical for understanding the weekly microcycle:
    - 0: Match day (peak readiness needed)
    - 1: Day before match (typically light session / activation)
    - 2-3: Mid-week (higher training loads expected)
    - 4-5: Early week post-match (recovery → progressive loading)

    This is an ENGINEERED FEATURE — not present in the raw data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'Player ID', 'Date', and 'Activity Type Today' columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with new 'Days Until Match' column (float, NaN if no future match).
    """
    print("\nAdding Days Until Match...")

    df_sorted = df.sort_values(['Player ID', 'Date']).copy()

    def calc_days_until_match(group):
        """
        Inner function applied per player group.

        For each row, find the nearest FUTURE date where Activity Type Today
        is exactly 'Game' (OHL first-team match only), and compute the days
        difference. Excludes 'Youth training or game' and 'National team
        training/game'.
        """
        # Find dates where the player has a game scheduled as Activity Type Today
        is_game_today = group['Activity Type Today'].str.strip().str.lower() == 'game'
        game_dates = group.loc[is_game_today, 'Date']

        result = pd.Series(index=group.index, dtype=float)
        for idx, row in group.iterrows():
            current_date = row['Date']
            # Find future (or same-day) games
            future_games = game_dates[game_dates >= current_date]
            if len(future_games) > 0:
                result[idx] = (future_games.min() - current_date).days
            else:
                result[idx] = np.nan
        return result

    # Apply per-player
    df_sorted['Days Until Match'] = df_sorted.groupby('Player ID', group_keys=False).apply(calc_days_until_match, include_groups=False)

    # Map back to original DataFrame order
    df_sorted['_temp_key'] = df_sorted['Player ID'].astype(str) + '_' + df_sorted['Date'].astype(str)
    df['_temp_key'] = df['Player ID'].astype(str) + '_' + df['Date'].astype(str)
    days_mapping = df_sorted.set_index('_temp_key')['Days Until Match'].to_dict()
    df['Days Until Match'] = df['_temp_key'].map(days_mapping)
    df = df.drop(columns=['_temp_key'])

    n_valid = df['Days Until Match'].notna().sum()
    print(f"  Added Days Until Match ({n_valid:,} valid entries)")

    return df


# =============================================================================
# STATUS FEATURES
# =============================================================================

def add_status_decrease(df):
    """
    Add binary flag for status decrease — THE PRIMARY PREDICTION TARGET.

    This is the most important engineered feature in the entire pipeline.
    It captures when a player's medical status WORSENED compared to their
    previous observation, which is what the ML models are trained to predict.

    === HOW IT WORKS (step by step) ===

    1. SORT the data by Player ID and Date so each player's timeline is
       in chronological order.

    2. SHIFT the 'Status' column forward by 1 within each player group
       using groupby('Player ID')['Status'].shift(1). This creates a
       'Previous Status' column where each row now knows what the player's
       status was on their PREVIOUS observation day.

       Example for Player 5:
           Date        Status      Previous Status
           Jan 10      Available   NaN              (first row → no previous)
           Jan 11      Available   Available
           Jan 12      Attention   Available        ← this is a DECREASE
           Jan 13      Injured     Attention        ← this is a DECREASE
           Jan 14      Injured     Injured          ← same status, NOT a decrease

    3. COMPARE Previous Status → Current Status. A status decrease is
       flagged (= 1) ONLY for these specific transitions:

       Previous Status    →    Current Status    =    Status Decrease
       ─────────────────────────────────────────────────────────────
       Available          →    Attention          =    1 (DECREASE)
       Available          →    Injured            =    1 (DECREASE)
       Attention          →    Injured            =    1 (DECREASE)
       ─────────────────────────────────────────────────────────────
       Injured            →    Attention          =    0 (improvement)
       Injured            →    Available          =    0 (improvement)
       Attention          →    Available          =    0 (improvement)
       Available          →    Available          =    0 (stable)
       Attention          →    Attention          =    0 (stable)
       Injured            →    Injured            =    0 (stable)
       NaN                →    anything           =    0 (first observation)

    4. MAP the result back to the original DataFrame order using the
       _temp_key pattern.

    === WHY THESE SPECIFIC TRANSITIONS? ===

    The status levels form a severity hierarchy:
        Available (healthy) < Attention (concern) < Injured (unable to play)

    We only flag DOWNWARD transitions (worsening). Improvements and stable
    states are labeled 0, making this a binary classification target.

    === CLASS IMBALANCE ===

    Status decreases are RARE events (typically <5% of observations).
    This creates severe class imbalance, which is why the ML models use
    techniques like class_weight='balanced' (LogReg), scale_pos_weight
    (XGBoost), and optimal threshold selection (Youden's J statistic).

    This is an ENGINEERED FEATURE — not present in the raw data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'Player ID', 'Date', and 'Status' columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with new 'Status Decrease' column (binary: 0 or 1).
    """
    print("\nAdding Status Decrease...")

    # Define the set of (previous, current) status pairs that count as a decrease.
    # Only transitions moving DOWN the severity hierarchy are included.
    # Using a set of tuples for O(1) lookup performance.
    decrease_transitions = {
        ('available', 'attention'),   # Healthy player flagged for concern
        ('available', 'injured'),     # Healthy player directly injured (severe)
        ('attention', 'injured')      # Concerned player progresses to injured
    }

    # Sort chronologically within each player to ensure correct temporal order
    df_sorted = df.sort_values(['Player ID', 'Date']).copy()

    # Get each player's status from their PREVIOUS observation day.
    # shift(1) within groupby ensures:
    #   - Only the SAME player's history is used (no cross-player leakage)
    #   - The first observation for each player gets NaN (no previous data)
    df_sorted['Previous Status'] = df_sorted.groupby('Player ID')['Status'].shift(1)

    # Compare previous vs current status for each row
    def is_decrease(row):
        """Check if the status transition represents a worsening."""
        # If either status is missing, we can't determine a transition → 0
        if pd.isna(row['Previous Status']) or pd.isna(row['Status']):
            return 0
        # Lowercase comparison for robustness against case inconsistencies
        prev = str(row['Previous Status']).lower()
        curr = str(row['Status']).lower()
        # Check if this (previous → current) pair is in our decrease set
        return 1 if (prev, curr) in decrease_transitions else 0

    # Apply the comparison to every row
    df_sorted['Status Decrease'] = df_sorted.apply(is_decrease, axis=1)

    # Map the computed values back to the original DataFrame's row order
    # using the _temp_key pattern (Player_ID + Date concatenation)
    df_sorted['_temp_key'] = df_sorted['Player ID'].astype(str) + '_' + df_sorted['Date'].astype(str)
    df['_temp_key'] = df['Player ID'].astype(str) + '_' + df['Date'].astype(str)

    decrease_mapping = df_sorted.set_index('_temp_key')['Status Decrease'].to_dict()
    df['Status Decrease'] = df['_temp_key'].map(decrease_mapping)

    # Clean up temporary column
    df = df.drop(columns=['_temp_key'])

    n_decrease = df['Status Decrease'].sum()
    print(f"  Added Status Decrease ({int(n_decrease):,} status decreases detected)")

    return df


# =============================================================================
# ACWR DANGER ZONE FEATURE
# =============================================================================

def add_any_acwr_danger(df):
    """
    Add binary flag indicating if ANY ACWR metric exceeds the danger zone (>1.5).

    The Acute:Chronic Workload Ratio (ACWR) compares a player's recent
    (7-day acute) training load to their longer-term (42-day chronic) load.
    This ratio indicates whether a player has recently spiked their workload
    relative to what their body is conditioned for.

    ACWR Interpretation (based on Gabbett 2016):
    - 0.8–1.3: "Sweet spot" — load matches conditioning, low injury risk
    - 1.3–1.5: Moderate risk — load slightly elevated
    - >1.5:    DANGER ZONE — load significantly exceeds conditioning,
               substantially elevated injury risk

    This function checks FOUR ACWR metrics:
    1. Total Distance
    2. High Speed Distance (>19.8 km/h)
    3. High Decelerations (>3 m/s²)
    4. Sprints (>25.2 km/h)

    If ANY of the four exceeds 1.5, the flag is set to 1.
    This provides a single "alarm" indicator without requiring the model
    to learn the 1.5 threshold from raw ACWR values.

    Reference: Gabbett, T. J. (2016). The training-injury prevention paradox.
    British Journal of Sports Medicine, 50(5), 273-280.

    This is an ENGINEERED FEATURE — not present in the raw data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ACWR columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with new 'Any ACWR Danger' column (binary: 0 or 1).
    """
    print("\nAdding Any ACWR Danger...")

    acwr_cols = [
        'Total Distance (ACWR) Yesterday',
        'High Speed Distance (ACWR) Yesterday',
        'High Decelerations (ACWR) Yesterday',
        'Sprints (ACWR) Yesterday'
    ]

    # Only use columns that actually exist in the data
    acwr_cols = [c for c in acwr_cols if c in df.columns]

    # Create a boolean DataFrame: True (1) where each ACWR > 1.5 danger threshold
    danger_flags = pd.DataFrame()
    for col in acwr_cols:
        danger_flags[col] = (df[col] > 1.5).astype(int)

    # max(axis=1): if ANY column is 1 → the row is 1 (logical OR across columns)
    df['Any ACWR Danger'] = danger_flags.max(axis=1)

    n_danger = df['Any ACWR Danger'].sum()
    print(f"  Added Any ACWR Danger ({n_danger:,} observations in danger zone)")

    return df


# =============================================================================
# COLUMN REORDERING
# =============================================================================

def reorder_columns(df):
    """
    Reorder columns into logical groups that reflect the TEMPORAL ORDERING
    of information availability within a single day.

    === TEMPORAL ORDERING WITHIN A ROW (date t) ===

    Each row represents one player-day. The columns are grouped by WHEN the
    information becomes available during that day:

    GROUP 1 — Identifiers (always known):
        Date, Playerkey, Player ID, Position

    GROUP 2 — Historical context (known before day t starts):
        Medical Availability Last 14 Days, Club Attendance Last 14 Days

    GROUP 3 — Yesterday's (t-1) data (known at start of day t):
        ACWR values, Activity Type Yesterday, Comments, GPS %, RPE
        These are PAST DATA — fully observed before any decisions on day t.

    GROUP 4 — Today's (t) morning assessment:
        Status, Status Decrease, Wellness z-scores, Composite scores,
        Days Since Game, Days Until Match
        These are measured before any activity happens on day t.

    GROUP 5 — Today's (t) post-assessment data (known AFTER morning assessment):
        Activity Type Today
        The session type assigned after the morning assessment. This is
        derived from the next row's Activity Type Yesterday (t+1 data),
        so it is temporally AFTER all other variables in the row.

    Any columns not explicitly listed are appended at the end.
    """
    print("\nReordering columns...")

    column_order = [
        # --- GROUP 1: Identifiers ---
        'Date',
        'Playerkey',
        'Player ID',
        'Position',
        # --- GROUP 2: Historical Context ---
        'Medical Availability Last 14 Days',
        'Club Attendance Last 14 Days',
        # --- GROUP 3: Yesterday (t-1) data — known at start of day t ---
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
        'Perceived Exertion Yesterday',
        # --- GROUP 4: Today (t) morning assessment — COVARIATES (X) ---
        'Status',
        'Status Decrease',
        'Fatigue (z)',
        'Readiness (z)',
        'Soreness (z)',
        'Physical State',
        'Sleep Quality (z)',
        'Stress (z)',
        'Mood (z)',
        'Mental State',
        'Overall Wellbeing',
        'Days Since Game',          # Today (t): days since last completed match (min=1, never 0)
        'Days Until Match',         # Today (t): days until next scheduled match (0 on match day)
        # --- GROUP 5: Today (t) post-assessment — derived from t+1 data ---
        'Activity Type Today',      # Session type assigned after morning assessment
        'Match Day',                # Team-level: 1 if any player has a game today, 0 otherwise
        'Selected',                 # Player-level: 1 if selected for match, 0 if not, NaN if no match
    ]
    
    available_columns = [col for col in column_order if col in df.columns]
    remaining_columns = [col for col in df.columns if col not in available_columns]
    final_order = available_columns + remaining_columns
    
    df = df[final_order]
    
    print(f"  Reordered {len(final_order)} columns")
    
    return df


# =============================================================================
# DATA SAVING
# =============================================================================

def save_processed_data(df):
    """Save processed data to CSV in the data/processed/ directory."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    output_path = project_root / "data" / "processed" / "Readiness_Data.csv"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"\nSaved processed data to: {output_path}")
    print(f"Final shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    return output_path


# =============================================================================
# PDF DOCUMENTATION GENERATION
# =============================================================================

def generate_documentation_pdf(df, player_mapping, output_dir):
    """
    Auto-generate a color-coded PDF data dictionary for the processed dataset.

    Creates a document that describes every column in the processed dataset,
    organized by temporal category with color-coding:
    - Green: General information (Date, Player ID, Position)
    - Blue:  Previous status variables (t-1: workload, activity, comments)
    - Orange: Current status variables (t: wellness, status, composites, post-assessment)

    Engineered features (created by this pipeline) are underlined to
    distinguish them from original raw data columns.

    Also includes a definitions section explaining key terms (ACWR, EMA,
    z-scores, personal benchmarks, status decrease).

    Uses ReportLab library for PDF generation.
    """
    print("\nGenerating PDF documentation...")

    pdf_path = output_dir / "Processed Data Dictionary.pdf"
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=0.6*inch,
        rightMargin=0.6*inch,
        topMargin=0.6*inch,
        bottomMargin=0.6*inch
    )

    elements = []
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#1a1a2e'),
        spaceAfter=10,
        fontName='Helvetica-Bold',
        alignment=TA_CENTER
    )

    overview_style = ParagraphStyle(
        'Overview',
        parent=styles['Normal'],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#333333'),
        spaceAfter=12,
        alignment=TA_CENTER
    )

    cell_style = ParagraphStyle(
        'CellText',
        parent=styles['Normal'],
        fontSize=8,
        leading=10,
        textColor=colors.black
    )

    var_style_white = ParagraphStyle(
        'VarNameWhite',
        parent=styles['Normal'],
        fontSize=8,
        leading=10,
        fontName='Helvetica-Bold',
        textColor=colors.white
    )

    # Title
    elements.append(Paragraph("Data Dictionary — Processed Dataset", title_style))

    # Overview sentence
    elements.append(Paragraph(
        "Processed dataset after feature engineering. Each row represents one player-day. "
        "Variables are grouped by their temporal position within the day: "
        "general identifiers, previous status (t-1), and current status (t).",
        overview_style
    ))

    # Legend with colored boxes (3 colors — matches raw data dictionary style)
    legend_data = [
        [
            Paragraph('<font color="#4CAF50"><b>||</b></font> General', styles['Normal']),
            Paragraph('<font color="#2196F3"><b>||</b></font> Previous Status (t-1)', styles['Normal']),
            Paragraph('<font color="#FF9800"><b>||</b></font> Current Status (t)', styles['Normal']),
        ]
    ]
    legend_table = Table(legend_data, colWidths=[2.2*inch]*3)
    legend_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elements.append(legend_table)
    elements.append(Spacer(1, 0.15*inch))

    # Color definitions (3 colors only — no separate treatment color)
    color_general = colors.HexColor('#4CAF50')
    color_previous = colors.HexColor('#2196F3')
    color_current = colors.HexColor('#FF9800')

    # Variable definitions: [variable_name, description, is_engineered]
    general_vars = [
        ['Date', 'Date of the observation', False],
        ['Playerkey', 'Unique anonymized player identifier', False],
        ['Player ID', 'Sequential player ID (1, 2, 3, ...) for easier reference', True],
        ['Position', 'Playing position (CD, ST, CDM, CAM, FB, WG, WB)', False],
    ]

    previous_vars = [
        ['Medical Availability\nLast 14 Days', 'Medical availability (%) over the last 14 days', False],
        ['Club Attendance\nLast 14 Days', 'Club attendance (%) over the last 14 days', False],
        ['Total Distance\n(ACWR) Yesterday', 'Total Distance — ACWR (7-day acute / 42-day chronic, EMA)', False],
        ['High Speed Distance\n(ACWR) Yesterday', 'High-Speed Distance (>19.8 km/h) — ACWR', False],
        ['High Decelerations\n(ACWR) Yesterday', 'High decelerations (>3 m/s2) — ACWR', False],
        ['Sprints (ACWR)\nYesterday', 'Sprint count (>25.2 km/h) — ACWR', False],
        ['Any ACWR Danger', 'Binary flag: 1 if any ACWR > 1.5 (danger zone)', True],
        ['Activity Type\nYesterday', 'Activity type from previous day (Training, Game, Rehab, etc.)', False],
        ['Comment Yesterday', 'Staff notes on previous day status', False],
        ['Comment Category\nYesterday', 'Categorized comment (recovery, discomfort, stiffness, etc.)', True],
        ['Total Distance %\nYesterday', 'Total Distance as % of personal benchmark (5 best matches)', False],
        ['High Speed Distance\n% Yesterday', 'High-Speed Distance as % of personal benchmark', False],
        ['High Decelerations\n% Yesterday', 'Decelerations (>3 m/s2) as % of personal benchmark', False],
        ['Sprints % Yesterday', 'Sprints (>25.2 km/h) as % of personal benchmark', False],
        ['Max Velocity %\nYesterday', 'Max velocity as % of personal best', False],
        ['Perceived Exertion\nYesterday', 'Rate of Perceived Exertion — z-score (based on 28 days)', False],
    ]

    current_vars = [
        ['Status', 'Current medical status (Available, Attention, Injured, Sick, Absent). Assessed in the morning.', False],
        ['Status Decrease', 'Binary: 1 if status worsened vs previous day (Available->Attention/Injured or Attention->Injured)', True],
        ['Fatigue (z)', 'Self-reported fatigue — z-score (28-day rolling window)', False],
        ['Readiness (z)', 'Self-reported readiness to train — z-score', False],
        ['Soreness (z)', 'Self-reported muscle soreness — z-score', False],
        ['Physical State', 'Composite: mean of Fatigue, Readiness, Soreness z-scores', True],
        ['Sleep Quality (z)', 'Self-reported sleep quality — z-score', False],
        ['Stress (z)', 'Self-reported stress level — z-score', False],
        ['Mood (z)', 'Self-reported mood — z-score', False],
        ['Mental State', 'Composite: mean of Sleep Quality, Stress, Mood z-scores', True],
        ['Overall Wellbeing', 'Composite: mean of all 6 wellbeing z-scores', True],
        ['Days Since Game', 'Days since last completed match (morning of day t). Min = 1, never 0.', True],
        ['Days Until Match', 'Days until next scheduled match. 0 on match day. NaN if no future match.', True],
        ['Activity Type Today', 'Session type on day t (Training, Game, Rehab, etc.). Assigned after morning assessment. Derived from the next row\'s Activity Type Yesterday (t+1 data).', True],
        ['Match Day', 'Team-level binary: 1 if any player has a game/match on this date, 0 otherwise. Same value for all players on a given date.', True],
        ['Selected', 'Player-level: 1 if selected for match, 0 if not selected (team has match but player does other activity). NaN on non-match days.', True],
    ]

    def build_var_row(name, desc, is_engineered):
        """Build a table row with proper text wrapping."""
        if is_engineered:
            name_para = Paragraph(f'<u>{name}</u>', var_style_white)
        else:
            name_para = Paragraph(name, var_style_white)

        desc_para = Paragraph(desc, cell_style)
        return [name_para, desc_para]

    # Build all rows with their colors
    all_rows = []
    row_colors = []

    for var in general_vars:
        all_rows.append(build_var_row(var[0], var[1], var[2]))
        row_colors.append(color_general)
    for var in previous_vars:
        all_rows.append(build_var_row(var[0], var[1], var[2]))
        row_colors.append(color_previous)
    for var in current_vars:
        all_rows.append(build_var_row(var[0], var[1], var[2]))
        row_colors.append(color_current)

    # Create table data (header + rows)
    table_data = [
        [
            Paragraph('<b>Variable</b>', ParagraphStyle('Header', fontSize=10, textColor=colors.white)),
            Paragraph('<b>Description</b>', ParagraphStyle('Header', fontSize=10, textColor=colors.white))
        ]
    ]

    for row in all_rows:
        table_data.append(row)

    # Create table
    var_table = Table(table_data, colWidths=[1.8*inch, 4.8*inch])

    # Build style commands
    style_commands = [
        # Header
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2d2d2d')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, 0), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('TOPPADDING', (0, 0), (-1, 0), 10),

        # All cells
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 1), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),

        # Description column background
        ('BACKGROUND', (1, 1), (1, -1), colors.HexColor('#fafafa')),
    ]

    # Add row-specific background colors for variable name column
    for row_idx, color in enumerate(row_colors, start=1):
        style_commands.append(('BACKGROUND', (0, row_idx), (0, row_idx), color))

    var_table.setStyle(TableStyle(style_commands))
    elements.append(var_table)

    # Note about underlined variables
    elements.append(Spacer(1, 0.15*inch))
    note_style = ParagraphStyle(
        'Note',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#555555'),
        alignment=TA_LEFT
    )
    elements.append(Paragraph(
        "<b>Note:</b> <u>Underlined variables</u> are engineered features "
        "created during preprocessing (not present in the raw data).",
        note_style
    ))

    # Spacer before definitions (no page break — keep on same page if possible)
    elements.append(Spacer(1, 0.25*inch))

    # Definitions section
    def_title_style = ParagraphStyle(
        'DefTitle',
        parent=styles['Heading2'],
        fontSize=12,
        textColor=colors.HexColor('#1a1a2e'),
        spaceBefore=6,
        spaceAfter=8,
        fontName='Helvetica-Bold'
    )
    elements.append(Paragraph("Definitions", def_title_style))

    definition_style = ParagraphStyle(
        'Definition',
        parent=styles['Normal'],
        fontSize=9,
        leading=12,
        spaceBefore=4,
        spaceAfter=4,
        textColor=colors.HexColor('#333333')
    )

    definitions = [
        ("<b>ACWR (Acute:Chronic Workload Ratio)</b> — Compares short-term "
         "(acute, 7-day) to long-term (chronic, 42-day) training load using EMA. "
         "Sweet spot: 0.8-1.3. Ratios >1.5 indicate elevated injury risk."),

        ("<b>EMA (Exponential Moving Average)</b> — Weights recent observations "
         "more heavily than older ones, providing a responsive measure of "
         "current fitness and fatigue states."),

        ("<b>Personal Benchmark</b> — Average of the player's five best match "
         "performances. Percentage metrics are relative to this individual benchmark."),

        ("<b>Z-scores</b> — Standardized scores where 0 = player's individual mean "
         "and +/-1 = one standard deviation. Based on 28-day rolling window per player."),

        ("<b>Status Decrease</b> — Binary indicator (0/1) flagging when a player's "
         "medical status worsened vs the previous day (Available->Attention, "
         "Available->Injured, or Attention->Injured)."),

        ("<b>Days Since Game</b> — Days since the player's last completed match. "
         "Measured in the morning before any activity. Minimum value is 1. Never 0."),

        ("<b>Days Until Match</b> — Days until the player's next scheduled match. "
         "Equals 0 on match day itself. Captures the weekly microcycle phase."),
    ]

    for definition in definitions:
        elements.append(Paragraph(definition, definition_style))

    # Build PDF
    doc.build(elements)
    print(f"  PDF documentation saved to: {pdf_path}")

    return pdf_path


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def preprocess_data():
    """
    Main preprocessing pipeline — orchestrates all transformation steps.

    Executes the full pipeline in order:
    1. Load raw data
    2. Basic transforms (player IDs, column renaming, percentage cleaning)
    3. Comment analysis (keyword categorization)
    4. Composite scores (Physical State, Mental State, Overall Wellbeing)
    5. Temporal features (Activity Type Today, Days Since Game)
    6. Status features (Status Decrease binary target)
    7. ACWR danger flag
    8. Column reordering
    9. Save to CSV + generate PDF documentation

    Returns
    -------
    pd.DataFrame
        The fully preprocessed DataFrame (4,240 rows × 33 columns).
    """
    print("=" * 80)
    print("OHL PLAYER READINESS - DATA PREPROCESSING PIPELINE")
    print("=" * 80)
    
    # Load data
    df = load_raw_data()
    
    # Basic Transformations
    df, player_mapping = map_player_ids(df)
    df = rename_columns(df)
    df = clean_percentage_columns(df)
    
    # Comment Analysis
    df = add_comment_category(df)
    
    # Composite Scores
    df = add_physical_state(df)
    df = add_mental_state(df)
    df = add_overall_wellbeing(df)
    
    # Temporal Features
    df = add_activity_type_today(df)
    df = add_days_since_game(df)
    df = add_days_until_match(df)

    # Match Day Features (depends on Activity Type Today)
    df = add_match_day_and_selected(df)

    # Status Features
    df = add_status_decrease(df)
    
    # ACWR Danger Flag
    df = add_any_acwr_danger(df)
    
    # Final Organization
    df = reorder_columns(df)
    
    # Save Results
    output_path = save_processed_data(df)
    
    # Generate Documentation
    generate_documentation_pdf(df, player_mapping, output_path.parent)
    
    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE")
    print(f"  Total features: {df.shape[1]}")
    print(f"  Total observations: {df.shape[0]:,}")
    print("=" * 80)
    
    return df


if __name__ == "__main__":
    preprocess_data()