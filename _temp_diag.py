import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(r"C:\Users\U0152019\PhD Documents\Projects\2. Readiness To Train\Readiness-To-Train")

# Load processed data
df = pd.read_excel(ROOT / "data/processed/RTT.xlsx")
df['Date'] = pd.to_datetime(df['Date'])

print("=== PROCESSED DATA OVERVIEW ===")
print(f"Shape: {df.shape}")
print(f"Columns with 'Activity': {[c for c in df.columns if 'Activity' in c]}")
print(f"Columns with 'Training Intensity': {[c for c in df.columns if 'Training Intensity' in c]}")
print()

# Activity Type Yesterday value counts
print("=== Activity Type Yesterday value counts ===")
print(df['Activity Type Yesterday'].value_counts(dropna=False))
print()

# GROUP A: NaN Activity Type Yesterday but TI Yesterday is NOT NaN
group_a = df[df['Activity Type Yesterday'].isna() & df['Training Intensity Yesterday'].notna()]
print(f"=== GROUP A: NaN Activity Type Yesterday BUT TI Yesterday is present ===")
print(f"Count: {len(group_a)}")
if len(group_a) > 0:
    print(f"TI Yesterday values: min={group_a['Training Intensity Yesterday'].min():.3f}, max={group_a['Training Intensity Yesterday'].max():.3f}, mean={group_a['Training Intensity Yesterday'].mean():.3f}")
    print(f"Players: {sorted(group_a['Player ID'].unique().tolist())}")
    print(f"Date range: {group_a['Date'].min()} to {group_a['Date'].max()}")
    # Check what Activity Type Today looks like for these rows
    if 'Activity Type Today' in df.columns:
        print(f"Activity Type Today distribution:")
        print(group_a['Activity Type Today'].value_counts(dropna=False))
    print(f"Status distribution:")
    print(group_a['Status'].value_counts(dropna=False))
    print(f"Sample rows (Date, Player ID, ATY, TIY, Status, DUM, DSG):")
    cols_show = [c for c in ['Date', 'Player ID', 'Activity Type Yesterday', 'Training Intensity Yesterday', 
                              'Status', 'Days Until Match', 'Days Since Game', 
                              'Total Distance (ACWR) Yesterday', 'Fatigue (z)'] if c in df.columns]
    print(group_a[cols_show].head(20).to_string())
print()

# GROUP B: Activity Type Yesterday = 'Training' but TI Yesterday IS NaN
group_b = df[df['Activity Type Yesterday'] == 'Training']
group_b_missing = group_b[group_b['Training Intensity Yesterday'].isna()]
print(f"=== GROUP B: Activity Type Yesterday = 'Training' BUT TI Yesterday IS NaN ===")
print(f"Total Training rows: {len(group_b)}")
print(f"Training rows with missing TI: {len(group_b_missing)} ({100*len(group_b_missing)/len(group_b):.1f}%)")
if len(group_b_missing) > 0:
    print(f"Players: {sorted(group_b_missing['Player ID'].unique().tolist())}")
    print(f"Date range: {group_b_missing['Date'].min()} to {group_b_missing['Date'].max()}")
    print(f"Status distribution:")
    print(group_b_missing['Status'].value_counts(dropna=False))
    # Check ACWR and raw GPS columns
    acwr_cols = [c for c in df.columns if 'ACWR' in c and 'Danger' not in c]
    raw_gps_cols = [c for c in df.columns if 'Yesterday' in c and any(x in c for x in ['Distance (m)', 'Minutes', 'Heart Rate'])]
    print(f"ACWR columns presence (non-null count for missing-TI rows):")
    for col in acwr_cols:
        print(f"  {col}: {group_b_missing[col].notna().sum()} non-null")
    print(f"Raw GPS columns presence (non-null count for missing-TI rows):")
    for col in raw_gps_cols:
        print(f"  {col}: {group_b_missing[col].notna().sum()} non-null")
    cols_show = [c for c in ['Date', 'Player ID', 'Activity Type Yesterday', 'Training Intensity Yesterday',
                              'Total Distance (ACWR) Yesterday', 'Total Distance (m) Yesterday',
                              'Status', 'Days Until Match', 'Days Since Game'] if c in df.columns]
    print(f"Sample rows:")
    print(group_b_missing[cols_show].head(20).to_string())
print()

# === VERIFICATION: Cross-check with raw Sessions.xlsx ===
print("=== RAW DATA VERIFICATION ===")
sessions = pd.read_excel(ROOT / "data/raw/Sessions.xlsx")
sessions['Date_Value'] = pd.to_datetime(sessions['Date_Value'])
print(f"Sessions.xlsx shape: {sessions.shape}")
print(f"Sessions columns: {sessions.columns.tolist()}")
print(f"Date range: {sessions['Date_Value'].min()} to {sessions['Date_Value'].max()}")
print(f"Session types: {sessions['session_type'].value_counts().to_dict() if 'session_type' in sessions.columns else 'N/A'}")
print(f"Reason values: {sessions['Reason'].value_counts().to_dict() if 'Reason' in sessions.columns else 'N/A'}")
print()

# Load Readiness_Data1 raw
rd1 = pd.read_excel(ROOT / "data/raw/Readiness_Data1.xlsx")
rd1.columns = [str(c) for c in rd1.columns]
print(f"Readiness_Data1.xlsx shape: {rd1.shape}")
print(f"Columns: {rd1.columns.tolist()}")
rd1_date_col = [c for c in rd1.columns if 'Date' in c or 'date' in c][0]
rd1['Date'] = pd.to_datetime(rd1[rd1_date_col])
print(f"Date range: {rd1['Date'].min()} to {rd1['Date'].max()}")

# Check what column has 'Reason' or activity info in rd1
reason_cols = [c for c in rd1.columns if 'Reason' in c or 'reason' in c or 'Activity' in c]
print(f"Reason/Activity columns in Readiness_Data1: {reason_cols}")
if reason_cols:
    print(f"Reason value counts in Readiness_Data1:")
    print(rd1[reason_cols[0]].value_counts(dropna=False).head(20))
print()

# Now for NaN Activity Type Yesterday rows: what date does the YESTERDAY correspond to?
# Activity Type Yesterday in row for date t corresponds to the session on date t-1
# So we need to look at what sessions happened on t-1 for those dates

nan_aty_dates = df[df['Activity Type Yesterday'].isna()]['Date'].unique()
print(f"Number of unique dates where Activity Type Yesterday = NaN: {len(nan_aty_dates)}")
# The "yesterday" dates are one day before these
nan_yest_dates = pd.DatetimeIndex(nan_aty_dates) - pd.Timedelta(days=1)
print(f"Corresponding 'yesterday' dates range: {nan_yest_dates.min()} to {nan_yest_dates.max()}")

# Check if those yesterday dates appear in Sessions.xlsx
if 'Date_Value' in sessions.columns:
    sessions_dates = pd.DatetimeIndex(sessions['Date_Value'].dt.normalize().unique())
    nan_yest_in_sessions = [d for d in nan_yest_dates if d.normalize() in sessions_dates]
    print(f"Of {len(nan_yest_dates)} 'yesterday' dates with NaN Activity Type, {len(nan_yest_in_sessions)} appear in Sessions.xlsx")
    if nan_yest_in_sessions:
        print(f"These sessions are:")
        for d in nan_yest_in_sessions[:10]:
            sess = sessions[sessions['Date_Value'].dt.normalize() == d.normalize()]
            print(f"  Date: {d.date()}, Sessions: {sess[['session_type', 'Reason']].to_dict('records') if all(c in sess.columns for c in ['session_type','Reason']) else sess.to_dict('records')}")

# Check in Readiness_Data1 raw data
if reason_cols:
    rc = reason_cols[0]
    # For each nan_yest_date, check what's in rd1
    rd1_nan_yest = rd1[rd1['Date'].dt.normalize().isin([d.normalize() for d in nan_yest_dates])]
    print(f"\nIn Readiness_Data1, rows where date = NaN-Activity yesterday dates:")
    print(f"Count: {len(rd1_nan_yest)}")
    if len(rd1_nan_yest) > 0:
        print(f"Reason distribution:")
        print(rd1_nan_yest[rc].value_counts(dropna=False))
print()
print("=== DONE ===")
