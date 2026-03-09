import json
from pathlib import Path

ROOT = Path(r"C:\Users\U0152019\PhD Documents\Projects\2. Readiness To Train\Readiness-To-Train")

def code_cell(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src if isinstance(src, list) else [src]}

def md_cell(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src if isinstance(src, list) else [src]}

cells = []

# ── TITLE ──
cells.append(md_cell([
    "# Temporary: Training Intensity Anomaly Analysis\n",
    "\n",
    "Two targeted investigations into unexpected patterns in `Training Intensity Yesterday`:\n",
    "\n",
    "- **Group A**: `Activity Type Yesterday` is **NaN** but `Training Intensity Yesterday` is **present** — how? what values?\n",
    "- **Group B**: `Activity Type Yesterday` = **Training** but `Training Intensity Yesterday` is **NaN** — why? what circumstances?\n",
    "\n",
    "Also includes cross-verification of NaN activity days against the raw `Sessions.xlsx` and `Readiness_Data1.xlsx` files."
]))

# ── SETUP ──
cells.append(md_cell("## 0. Setup"))
cells.append(code_cell([
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib.gridspec as gridspec\n",
    "import seaborn as sns\n",
    "from pathlib import Path\n",
    "\n",
    "ROOT = Path(r'C:\Users\U0152019\PhD Documents\Projects\2. Readiness To Train\Readiness-To-Train')\n",
    "\n",
    "# Load processed data\n",
    "df = pd.read_excel(ROOT / 'data/processed/RTT.xlsx')\n",
    "df['Date'] = pd.to_datetime(df['Date'])\n",
    "\n",
    "# Load raw data for verification\n",
    "sessions = pd.read_excel(ROOT / 'data/raw/Sessions.xlsx')\n",
    "sessions['Date_Value'] = pd.to_datetime(sessions['Date_Value'])\n",
    "\n",
    "rd1 = pd.read_excel(ROOT / 'data/raw/Readiness_Data1.xlsx')\n",
    "date_col = [c for c in rd1.columns if 'date' in str(c).lower()][0]\n",
    "rd1['Date'] = pd.to_datetime(rd1[date_col])\n",
    "\n",
    "# Define masks\n",
    "mask_nan_aty   = df['Activity Type Yesterday'].isna()\n",
    "mask_ti_present = df['Training Intensity Yesterday'].notna()\n",
    "mask_ti_missing = df['Training Intensity Yesterday'].isna()\n",
    "mask_training_aty = df['Activity Type Yesterday'] == 'Training'\n",
    "\n",
    "# Group A: NaN Activity Type Yesterday BUT TI present\n",
    "grp_a = df[mask_nan_aty & mask_ti_present].copy()\n",
    "\n",
    "# Group B: Activity Type Yesterday = Training BUT TI missing\n",
    "grp_b = df[mask_training_aty & mask_ti_missing].copy()\n",
    "\n",
    "print(f'Processed data: {df.shape}')\n",
    "print(f'Group A (NaN ATY + TI present): {len(grp_a)} rows ({100*len(grp_a)/len(df):.1f}%)')\n",
    "print(f'Group B (Training ATY + TI missing): {len(grp_b)} rows ({100*len(grp_b)/len(df[mask_training_aty]):.1f}% of all Training rows)')\n",
    "print(f'Sessions.xlsx: {sessions.shape}')\n",
    "print(f'Readiness_Data1.xlsx: {rd1.shape}')"
]))

# ── SECTION 1: RAW DATA VERIFICATION ──
cells.append(md_cell([
    "## 1. Raw Data Verification: Do NaN Activity Days Have No Session?\n",
    "\n",
    "We check the raw `Readiness_Data1.xlsx` `Reason` column and `Sessions.xlsx` to verify\n",
    "whether NaN `Activity Type Yesterday` rows correspond to days with truly no recorded session."
]))
cells.append(code_cell([
    "# The 'yesterday' date for a row with date t is t-1\n",
    "nan_aty_rows = df[mask_nan_aty].copy()\n",
    "nan_yest_dates = (nan_aty_rows['Date'] - pd.Timedelta(days=1)).dt.normalize().unique()\n",
    "\n",
    "print(f'Rows with NaN Activity Type Yesterday: {len(nan_aty_rows)}')\n",
    "print(f'Unique player-days behind those (yesterday dates): {len(nan_yest_dates)}')\n",
    "\n",
    "# Check Sessions.xlsx\n",
    "sessions['date_norm'] = sessions['Date_Value'].dt.normalize()\n",
    "session_dates = set(sessions['date_norm'].unique())\n",
    "nan_yest_in_sessions = [d for d in nan_yest_dates if pd.Timestamp(d) in session_dates]\n",
    "print(f'Of those yesterday dates, {len(nan_yest_in_sessions)} appear in Sessions.xlsx (team-level sessions existed)')\n",
    "\n",
    "# Check Readiness_Data1 Reason column\n",
    "reason_col = [c for c in rd1.columns if 'reason' in str(c).lower() or 'Reason' in str(c)]\n",
    "reason_col = reason_col[0] if reason_col else None\n",
    "if reason_col:\n",
    "    rd1_nan_days = rd1[rd1['Date'].dt.normalize().isin(nan_yest_dates)]\n",
    "    print(f'\nIn Readiness_Data1, rows on those dates: {len(rd1_nan_days)}')\n",
    "    print('Reason distribution:')\n",
    "    print(rd1_nan_days[reason_col].value_counts(dropna=False).to_string())\n",
    "    print('\nConclusion: NaN in Activity Type Yesterday comes from NaN in raw Reason column,')\n",
    "    print('not from an absence of a session. GPS % data can still be present on those rows.')\n",
    "\n",
    "# Show the session types on dates that DID have a session in Sessions.xlsx\n",
    "if nan_yest_in_sessions:\n",
    "    overlapping_sessions = sessions[sessions['date_norm'].isin(nan_yest_in_sessions)]\n",
    "    print(f'\nSession types on those {len(nan_yest_in_sessions)} dates (from Sessions.xlsx):')\n",
    "    if 'session_type' in sessions.columns:\n",
    "        print(overlapping_sessions['session_type'].value_counts().to_string())\n",
    "    if 'Reason' in sessions.columns:\n",
    "        print('Reason:')\n",
    "        print(overlapping_sessions['Reason'].value_counts().to_string())"
]))

# ── SECTION 2: GROUP A OVERVIEW ──
cells.append(md_cell([
    "## 2. Group A: NaN Activity Type Yesterday — but TI Yesterday is Present\n",
    "\n",
    "**Explanation:** `Activity Type Yesterday` is derived from the `Reason` column in `Readiness_Data1`.\n",
    "When `Reason` is blank, `Activity Type Yesterday` is NaN — but the GPS % columns (TD%, HSD%, Dec%, Sprints%)\n",
    "used to compute `Training Intensity Yesterday` can still be filled in the same row.\n",
    "So TI is computable even when no activity label was recorded."
]))
cells.append(code_cell([
    "print(f'Group A size: {len(grp_a)} rows across {grp_a[\"Player ID\"].nunique()} players')\n",
    "print(f'\nTI Yesterday values:')\n",
    "print(f'  min={grp_a[\"Training Intensity Yesterday\"].min():.3f}')\n",
    "print(f'  median={grp_a[\"Training Intensity Yesterday\"].median():.3f}')\n",
    "print(f'  mean={grp_a[\"Training Intensity Yesterday\"].mean():.3f}')\n",
    "print(f'  max={grp_a[\"Training Intensity Yesterday\"].max():.3f}')\n",
    "\n",
    "# Distribution of TI values\n",
    "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n",
    "\n",
    "# TI distribution comparison\n",
    "ti_all_valid = df[df['Training Intensity Yesterday'].notna()]['Training Intensity Yesterday']\n",
    "ti_grp_a = grp_a['Training Intensity Yesterday']\n",
    "\n",
    "axes[0].hist(ti_all_valid, bins=40, alpha=0.6, label='All valid TI rows', color='steelblue', density=True)\n",
    "axes[0].hist(ti_grp_a, bins=30, alpha=0.7, label='Group A (NaN Activity Type)', color='darkorange', density=True)\n",
    "axes[0].set_xlabel('Training Intensity Yesterday')\n",
    "axes[0].set_ylabel('Density')\n",
    "axes[0].set_title('TI Distribution: All Valid vs Group A')\n",
    "axes[0].legend()\n",
    "\n",
    "# Activity Type Today on Group A rows\n",
    "if 'Activity Type Today' in grp_a.columns:\n",
    "    att_counts = grp_a[
