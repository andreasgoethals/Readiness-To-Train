"""
Data Loader for Readiness Prediction Models
Handles loading, lagging, temporal splitting, and preprocessing of data.

Takes the processed dataset (RTT.xlsx) and creates ML-ready train/val/test
splits with all the transformations needed for model training.

=== CAPABILITIES ===

1. VARIABLE SELECTION
   - Covariates (X): specified via `predictory_columns` — the features for prediction
   - Treatment (T): specified via `treatment_columns` — optional, for causal ML methods.
     Any column(s) can be designated as treatment; the choice depends on the experiment.
   - Outcome (Y): specified via `target_variable` — the variable to predict

2. TEMPORAL ALIGNMENT (timeline mechanics)
   Each row in the processed dataset has data from different time points:
       - "Yesterday" columns = data from t-1
       - Morning assessment columns = data from t (wellness, status)
       - Activity Type Today = data from t, but decided AFTER morning assessment

   The loader lets you control WHICH ROW each variable comes from:

   `target_horizon` (int, default=0):
       Shifts the target variable forward N steps within each player.
       - target_horizon=0: outcome from the SAME row as covariates
       - target_horizon=1: outcome from the NEXT row (predict tomorrow)

   `treatment_horizon` (int or dict, default=0):
       Shifts treatment columns forward N steps within each player.
       - treatment_horizon=0: treatment from the SAME row as covariates
       - treatment_horizon=1: treatment from the NEXT row
       - dict: per-column offsets for mixed alignment, e.g.:
         {'Activity Type Today': 0, 'Total Distance % Yesterday': 1}

3. LAG FEATURES
   `lag` (int, default=0): Creates lagged versions of each covariate.
       lag=0: only current-row values
       lag=3: adds _t-1, _t-2, _t-3 for each covariate
   All lag operations are player-grouped (no cross-player contamination).

4. MISSING DATA HANDLING
   - Numerical: 'mean', 'median', 'zero', 'forward_fill', 'drop'
   - Categorical: 'mode', 'constant', 'forward_fill', 'drop'
   All imputation statistics are computed from training data only.

5. CATEGORICAL ENCODING
   - 'one-hot': binary dummy columns (preferred for linear models)
   - 'label': integer mapping (preferred for tree-based models)
   Encoding is fitted on training data; unseen categories are handled safely.

6. STANDARDIZATION
   - `standardize=True`: z-score normalization of numerical covariates
   - `standardize_treatment`: separate control for treatment variables
   Scaler is fitted on training data only.

7. TEMPORAL SPLITTING (vs. random splitting)
   Uses date-block splitting: early dates → train, middle → val, late → test.
   This prevents same-day leakage and mimics real deployment where we train
   on past data and predict future outcomes.

=== ANTI-LEAKAGE PIPELINE ORDER ===

   1. Create lag features (BEFORE splitting — needs full player time series)
   2. Apply treatment and target horizon shifts (BEFORE splitting)
   3. SPLIT the data temporally (by date blocks)
   4. Impute missing values (fit on TRAIN ONLY, transform all)
   5. Encode categoricals  (fit on TRAIN ONLY, transform all)
   6. Standardize features  (fit on TRAIN ONLY, transform all)

=== USAGE EXAMPLES ===

Standard ML (covariates → outcome, no treatment separation):

    loader = ReadinessDataLoader()
    data = loader.create_dataset(
        target_variable='Status Decrease',
        predictory_columns=['Fatigue (z)', 'Readiness (z)', ...],
        lag=3,
        include_previous_target=True,
        test_size=0.2,
        val_size=0.1,
        categorical_encoding='one-hot',
        standardize=True
    )
    # Returns: X_train, y_train, X_test, y_test, ...

With treatment separation (for causal ML methods):

    data = loader.create_dataset(
        target_variable='Status Decrease',
        predictory_columns=['Fatigue (z)', 'Readiness (z)', ...],
        treatment_columns=['Activity Type Today'],
        treatment_horizon=0,
        target_horizon=1,
        lag=3,
        categorical_encoding='label'
    )
    # Returns: X_train, T_train, y_train, X_test, T_test, y_test, ...
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import warnings


class ReadinessDataLoader:
    """
    Data loader for readiness prediction models.

    Handles the complete data pipeline from processed data (RTT.xlsx) to
    ML-ready train/val/test splits with proper anti-leakage measures.

    Supports flexible variable selection:
    - Covariates (X): any columns via `predictory_columns`
    - Treatment (T): any columns via `treatment_columns` (optional, for causal ML)
    - Outcome (Y): any column via `target_variable`

    The loader stores fitted transformers (scaler, encoder) as instance
    variables so they can potentially be reused to transform new/unseen
    data using the same statistics learned from the training set.

    Attributes
    ----------
    data_path : Path
        Path to the processed data file (RTT.xlsx).
    df : pd.DataFrame or None
        Loaded DataFrame (set after _verify_and_load_data is called).
    categorical_features : list
        Names of categorical feature columns (set after create_dataset).
    numerical_features : list
        Names of numerical feature columns (set after create_dataset).
    scaler : StandardScaler
        Scikit-learn scaler fitted on training data (for z-score normalization).
    treatment_scaler : StandardScaler
        Separate scaler for treatment variables (independent from covariate scaler).
    one_hot_encoder : OneHotEncoder or None
        Scikit-learn encoder fitted on training data (if one-hot encoding used).
    """

    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the data loader.

        Parameters
        ----------
        data_path : str, optional
            Path to processed data file. If None, uses default location
            (data/processed/RTT.xlsx relative to project root).
        """
        if data_path is None:
            # Navigate from src/data/ → src/ → project_root/
            script_dir = Path(__file__).parent
            project_root = script_dir.parent.parent
            self.data_path = project_root / "data" / "processed" / "RTT.xlsx"
        else:
            self.data_path = Path(data_path)

        self.df = None
        self.categorical_features = []
        self.numerical_features = []
        # These will be fitted on training data and used to transform val/test
        self.scaler = StandardScaler()
        self.treatment_scaler = StandardScaler()  # Separate scaler for treatment
        self.one_hot_encoder = None

    def _verify_and_load_data(self) -> pd.DataFrame:
        """
        Verify processed data exists and load it.

        If the processed data file does not exist yet (e.g., first run),
        this method automatically triggers the preprocessing pipeline from
        data_preprocessing.py to generate it. This provides a seamless
        experience where users can call create_dataset() without manually
        running preprocessing first.

        Returns
        -------
        pd.DataFrame
            Loaded processed data with Date parsed as datetime.
        """
        if not self.data_path.exists():
            print(f"Processed data not found at: {self.data_path}")
            print("Running preprocessing pipeline...")
            import sys
            # Ensure data_preprocessing can be found regardless of execution context
            _parent = str(Path(__file__).parent)
            if _parent not in sys.path:
                sys.path.insert(0, _parent)
            from data_preprocessing import preprocess_data
            df = preprocess_data()
            print("Preprocessing complete. Data loaded.")
        else:
            # Support both .xlsx and .csv for backwards compatibility
            if str(self.data_path).endswith('.xlsx'):
                df = pd.read_excel(self.data_path, parse_dates=['Date'], engine='openpyxl')
            else:
                df = pd.read_csv(self.data_path, parse_dates=['Date'])

        return df

    def load_processed_data(self) -> pd.DataFrame:
        """
        Return the raw processed DataFrame (before ML splits or encoding).

        Use this to share the same in-memory data with DAGCreator so that
        both the causal structure (DAGCreator) and the ML pipeline
        (ReadinessDataLoader) operate on the same source. Example::

            loader = ReadinessDataLoader()
            processed_df = loader.load_processed_data()

            from src.methods.DAG_Creator import DAGCreator
            creator = DAGCreator(
                player_id=1,
                state_vars=['Fatigue (z)', 'Readiness (z)', 'Soreness (z)'],
                treatment_var='Training Intensity Score',
                outcome_var='Match Performance',
                player_df=processed_df,   # share the already-loaded data
            )

        Returns
        -------
        pd.DataFrame
            Full processed dataset (all players), as loaded from disk,
            with Date parsed as datetime.
        """
        return self._verify_and_load_data()

    def _merge_activity_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge rare Activity Type categories to reduce cardinality.

        The raw activity types include many low-frequency categories
        (e.g., 'National team training/game', 'Suspension') that would
        create sparse one-hot columns or unreliable label encodings.
        This merges them into broader groups:

        - National team training/game, Youth training or game → 'Team Activities'
        - Individual, Private, Sickness, Suspension, Medical → 'Individual/Other'

        The main categories (Training, Game, Rehab, Day off, etc.) are
        left unchanged as they have sufficient observations.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with activity type columns.

        Returns
        -------
        pd.DataFrame
            DataFrame with merged activity categories.
        """
        activity_mapping = {
            'National team training/game': 'Team Activities',
            'Youth training or game': 'Team Activities',
            'Individual': 'Individual/Other',
            'Private': 'Individual/Other',
            'Sickness': 'Individual/Other',
            'Suspension': 'Individual/Other',
            'Medical': 'Individual/Other'
        }

        # Apply mapping to both yesterday's and today's activity columns
        for col in ['Activity Type Yesterday', 'Activity Type Today']:
            if col in df.columns:
                df[col] = df[col].replace(activity_mapping)

        return df

    def _identify_feature_types(
        self,
        columns: List[str],
        df: pd.DataFrame,
        context: str = 'predictor'
    ) -> Tuple[List[str], List[str]]:
        """
        Classify columns as categorical or numerical.

        Uses a two-tier approach:
        1. Known categoricals: a hard-coded set of column names known to be
           categorical (e.g., Position, Activity Type). This is more reliable
           than dtype inference for these specific columns.
        2. Dtype fallback: for unknown columns, uses pandas dtype inspection.
           Object and category dtypes → categorical; everything else → numerical.

        This classification determines how each feature is handled downstream:
        - Categorical features: imputed with mode/constant, then label/one-hot encoded
        - Numerical features: imputed with mean/median/zero, then optionally standardized

        Parameters
        ----------
        columns : List[str]
            List of column names to classify.
        df : pd.DataFrame
            DataFrame used to inspect dtypes for unknown columns.
        context : str, default='predictor'
            Context for the classification. When 'predictor', warns if
            Activity Type Today is included (it's a post-assessment variable).
            When 'treatment', suppresses this warning (the user has
            explicitly chosen it as a treatment variable).

        Returns
        -------
        tuple of (List[str], List[str])
            - categorical: list of categorical column names
            - numerical: list of numerical column names
        """
        categorical = []
        numerical = []

        # Domain-specific knowledge: these columns are always categorical
        # regardless of how pandas infers their dtype
        known_categorical = {
            'Position', 'Activity Type Yesterday', 'Comment Category Yesterday',
            'Status', 'Activity Type Today', 'Comment Yesterday'
        }

        # Warn if Activity Type Today is used as a predictor — it is determined
        # AFTER the morning assessment, so it is not available at prediction time.
        # Suppress warning when it's being used as a treatment variable (context='treatment')
        if context == 'predictor' and 'Activity Type Today' in columns:
            warnings.warn(
                "WARNING: 'Activity Type Today' is included as a predictor column. "
                "This variable is determined AFTER the morning assessment (it is "
                "derived from the next row's Activity Type Yesterday). It is not "
                "available at the time of prediction. Consider removing it from "
                "predictory_columns. If you intend to use it as a treatment variable "
                "for causal analysis, use the treatment_columns parameter instead.",
                UserWarning
            )

        for col in columns:
            if col in known_categorical:
                categorical.append(col)
            elif df[col].dtype == 'object' or df[col].dtype.name == 'category':
                # Fallback: use dtype for columns not in the known set
                categorical.append(col)
            else:
                numerical.append(col)

        return categorical, numerical

    def _create_lag_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        lag: int,
        include_target: bool = False,
        target_variable: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Create lagged (time-shifted) features for each predictor column.

        For each predictor column and each lag step t (1 to lag), creates a
        new column '{column}_t-{t}' containing that column's value from t
        time steps ago for the SAME player.

        Example with lag=2 and column 'Fatigue (z)':
            Creates: 'Fatigue (z)_t-1' (yesterday's fatigue)
                     'Fatigue (z)_t-2' (two days ago fatigue)

        === ANTI-LEAKAGE: Player-Grouped Shifting ===

        The .shift(t) operation is performed WITHIN each player group using
        groupby('Player ID'). This ensures:
        - Player A's lagged values come ONLY from Player A's history
        - No cross-player contamination (Player B's data can't appear as
          Player A's lagged feature)
        - The first t rows for each player get NaN (no historical data yet)

        If include_target=True, also creates lagged versions of the target
        variable (e.g., 'Status Decrease_t-1'). This allows models to learn
        from whether a player had a status decrease recently.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame sorted by Player ID and Date.
        columns : List[str]
            Predictor columns to create lag features for.
        lag : int
            Number of lag steps (e.g., 3 creates t-1, t-2, t-3).
        include_target : bool
            Whether to also create lagged versions of the target variable.
        target_variable : str, optional
            Name of the target column (needed if include_target=True).

        Returns
        -------
        pd.DataFrame
            DataFrame with new lag columns appended. Number of new columns
            = len(columns) * lag + (lag if include_target else 0).
        """
        if lag < 1:
            return df

        # Ensure chronological order within each player before shifting
        df = df.sort_values(['Player ID', 'Date']).copy()

        # Create lagged features for each predictor column
        for col in columns:
            for t in range(1, lag + 1):
                lag_col_name = f"{col}_t-{t}"
                # shift(t) within player group: pulls value from t rows back
                # First t rows for each player will be NaN (no history yet)
                df[lag_col_name] = df.groupby('Player ID')[col].shift(t)

        # Optionally create lagged versions of the target variable
        # This lets models learn from recent status decrease history
        if include_target and target_variable and lag >= 1:
            for t in range(1, lag + 1):
                lag_col_name = f"{target_variable}_t-{t}"
                df[lag_col_name] = df.groupby('Player ID')[target_variable].shift(t)

        return df

    def _create_treatment_lag_features(
        self,
        df: pd.DataFrame,
        treatment_columns: List[str],
        treatment_lag: int
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Create lagged features for treatment columns.

        Similar to _create_lag_features but specifically for treatment variables.
        Useful for modeling sequential treatment effects (e.g., how the last
        3 days of training intensity affect the outcome).

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame sorted by Player ID and Date.
        treatment_columns : List[str]
            Treatment columns to create lag features for.
        treatment_lag : int
            Number of lag steps for treatment history.

        Returns
        -------
        tuple of (pd.DataFrame, List[str])
            DataFrame with treatment lag columns appended, and list of
            new treatment lag column names.
        """
        if treatment_lag < 1:
            return df, []

        df = df.sort_values(['Player ID', 'Date']).copy()
        treatment_lag_columns = []

        for col in treatment_columns:
            for t in range(1, treatment_lag + 1):
                lag_col_name = f"{col}_t-{t}"
                df[lag_col_name] = df.groupby('Player ID')[col].shift(t)
                treatment_lag_columns.append(lag_col_name)

        return df, treatment_lag_columns

    def _apply_treatment_horizon(
        self,
        df: pd.DataFrame,
        treatment_columns: List[str],
        treatment_horizon: Union[int, Dict[str, int]]
    ) -> pd.DataFrame:
        """
        Shift treatment columns by their horizon offset within player groups.

        When treatment_horizon > 0, pulls the treatment value from N rows
        AHEAD (future) into the current row. This is useful when the variable
        of interest is recorded in a different row than the covariates.

        For example, if a variable appears as "Yesterday" data in the next
        row (t+1), setting treatment_horizon=1 for that column aligns it
        with the current row's covariates.

        The shift is done WITHIN each player group using groupby('Player ID')
        to prevent cross-player contamination.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame sorted by Player ID and Date.
        treatment_columns : List[str]
            Treatment column names.
        treatment_horizon : int or dict
            If int: apply same offset to all treatment columns.
            If dict: map column names to individual offsets.
            Columns not in the dict default to offset 0 (no shift).

        Returns
        -------
        pd.DataFrame
            DataFrame with treatment columns shifted as specified.
            Rows where shifted values are NaN (at end of player series)
            will need to be dropped later.
        """
        df = df.sort_values(['Player ID', 'Date']).copy()

        if isinstance(treatment_horizon, dict):
            # Per-column horizon: each treatment column can have a different offset
            for col in treatment_columns:
                horizon = treatment_horizon.get(col, 0)  # Default to 0 if not specified
                if horizon != 0:
                    # shift(-N) pulls value from N rows AHEAD within same player
                    df[col] = df.groupby('Player ID')[col].shift(-horizon)
        elif treatment_horizon != 0:
            # Uniform horizon: apply same offset to all treatment columns
            for col in treatment_columns:
                df[col] = df.groupby('Player ID')[col].shift(-treatment_horizon)

        return df

    def _temporal_split(
        self,
        df: pd.DataFrame,
        test_size: float,
        val_size: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Perform date-block temporal split (NOT random split).

        === WHY DATE-BLOCK SPLITTING? ===

        Two key reasons for splitting by date blocks rather than randomly:

        1. TEMPORAL REALISM: In production, we train on historical data and
           predict future outcomes. Random splitting would mix past and future,
           giving unrealistically optimistic performance estimates.

        2. SAME-DAY LEAKAGE PREVENTION: On any given date, multiple players
           have observations. If we randomly split, some players from Jan 15
           might be in training while others are in test. The model could learn
           date-specific patterns (e.g., post-game effects) from training
           players and apply them to test players from the SAME date. Date-block
           splitting ensures ALL observations from a given date go to the same set.

        === HOW IT WORKS ===

        1. Get all unique dates and sort chronologically
        2. Split the DATE timeline (not the rows) by proportion:
           - First (1 - test_size - val_size) of dates → training
           - Next val_size of dates → validation
           - Last test_size of dates → test

        Example with 100 unique dates, test_size=0.2, val_size=0.1:
           Dates 1-70  → training  (70% of dates)
           Dates 71-80 → validation (10% of dates)
           Dates 81-100 → test     (20% of dates)

        Note: The number of ROWS in each set may not match these proportions
        exactly, because different dates may have different numbers of players.

        Parameters
        ----------
        df : pd.DataFrame
            Full DataFrame with 'Date' column.
        test_size : float
            Proportion of dates for the test set (e.g., 0.2 = last 20% of dates).
        val_size : float
            Proportion of dates for the validation set.

        Returns
        -------
        tuple of (pd.DataFrame, pd.DataFrame, pd.DataFrame)
            train_df, val_df, test_df — non-overlapping date blocks.
        """
        if test_size + val_size >= 1.0:
            raise ValueError(f"test_size ({test_size}) + val_size ({val_size}) must be < 1.0")

        # Sort all data chronologically
        df = df.sort_values('Date').reset_index(drop=True)
        unique_dates = sorted(df['Date'].unique())
        n_dates = len(unique_dates)

        # Calculate cutoff indices on the DATE timeline (not row indices)
        train_date_end = int(n_dates * (1 - test_size - val_size))
        val_date_end = int(n_dates * (1 - test_size))

        # Slice the date timeline into three non-overlapping blocks
        train_dates = unique_dates[:train_date_end]        # Earliest dates
        val_dates = unique_dates[train_date_end:val_date_end]  # Middle dates
        test_dates = unique_dates[val_date_end:]           # Most recent dates

        # Assign all rows from each date block to the corresponding set
        # .copy() to avoid SettingWithCopyWarning in downstream operations
        train_df = df[df['Date'].isin(train_dates)].copy()
        val_df = df[df['Date'].isin(val_dates)].copy()
        test_df = df[df['Date'].isin(test_dates)].copy()

        return train_df, val_df, test_df

    def _impute_numerical(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        numerical_cols: List[str],
        strategy: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
        """
        Impute (fill in) missing values for numerical features.

        === CRITICAL: FIT on TRAIN, TRANSFORM on ALL ===

        The fill values (mean, median) are computed ONLY from training data.
        The same values are then applied to validation and test sets. This
        prevents test data statistics from leaking into training.

        Available strategies:
        - 'mean':         Fill NaN with column mean (from training set)
        - 'median':       Fill NaN with column median (from training set)
        - 'zero':         Fill NaN with 0 (no fitting needed)
        - 'forward_fill': Carry last valid value forward; remaining NaN → 0
        - 'drop':         Remove rows with any NaN in numerical columns

        Columns with no missing values in training are skipped (optimization).

        Parameters
        ----------
        train_df, val_df, test_df : pd.DataFrame
            The three data splits.
        numerical_cols : List[str]
            Numerical columns to impute.
        strategy : str
            Imputation strategy name.

        Returns
        -------
        tuple of (pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict)
            The three imputed DataFrames and a dict logging which values were used.
        """
        # .copy() to avoid modifying the original DataFrames
        train_df = train_df.copy()
        val_df = val_df.copy()
        test_df = test_df.copy()
        imputation_info = {}  # Track what was done for transparency/reproducibility

        for col in numerical_cols:
            if col not in train_df.columns:
                continue

            # Check if any split has missing values for this column.
            # We must check all splits (not just train) because val/test may
            # have NaN even when train doesn't (e.g., lag features at the start
            # of a player's series that falls in val/test).
            has_any_missing = (
                train_df[col].isna().any() or
                (len(val_df) > 0 and val_df[col].isna().any()) or
                test_df[col].isna().any()
            )
            if not has_any_missing:
                continue

            if strategy == 'mean':
                # Compute mean from TRAINING set only
                fill_value = train_df[col].mean()
                imputation_info[col] = {'strategy': 'mean', 'value': float(fill_value)}
            elif strategy == 'median':
                # Compute median from TRAINING set only
                fill_value = train_df[col].median()
                imputation_info[col] = {'strategy': 'median', 'value': float(fill_value)}
            elif strategy == 'zero':
                # No statistics needed — just use 0
                fill_value = 0
                imputation_info[col] = {'strategy': 'zero', 'value': 0}
            elif strategy == 'forward_fill':
                # Carry last valid value forward WITHIN each player (prevents
                # cross-player contamination when rows are interleaved by date).
                # If still NaN after ffill (start of a player's series), fill with 0.
                train_df[col] = train_df.groupby('Player ID')[col].ffill().fillna(0)
                val_df[col] = val_df.groupby('Player ID')[col].ffill().fillna(0)
                test_df[col] = test_df.groupby('Player ID')[col].ffill().fillna(0)
                imputation_info[col] = {'strategy': 'forward_fill', 'value': None}
                continue  # Skip the fillna below (already handled)
            elif strategy == 'drop':
                imputation_info[col] = {'strategy': 'drop', 'value': None}
                continue  # Rows dropped after the loop
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            # Apply the SAME fill value (from training) to all three sets
            train_df[col] = train_df[col].fillna(fill_value)
            val_df[col] = val_df[col].fillna(fill_value)
            test_df[col] = test_df[col].fillna(fill_value)

        # If strategy is 'drop', remove rows with remaining NaN values
        if strategy == 'drop':
            cols_to_check = [c for c in numerical_cols if c in train_df.columns]
            train_df = train_df.dropna(subset=cols_to_check)
            val_df = val_df.dropna(subset=cols_to_check)
            test_df = test_df.dropna(subset=cols_to_check)

        return train_df, val_df, test_df, imputation_info

    def _impute_categorical(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        categorical_cols: List[str],
        strategy: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
        """
        Impute (fill in) missing values for categorical features.

        Same anti-leakage principle as numerical imputation: the fill value
        (mode = most frequent category) is computed from training data only.

        Available strategies:
        - 'mode':         Fill NaN with most frequent category (from training set).
                          Falls back to 'missing' if no mode can be computed.
        - 'constant':     Fill NaN with the string 'missing' (explicit missing category).
        - 'forward_fill': Carry last valid category forward; remaining NaN → 'missing'.
        - 'drop':         Remove rows with any NaN in categorical columns.

        Parameters
        ----------
        train_df, val_df, test_df : pd.DataFrame
            The three data splits.
        categorical_cols : List[str]
            Categorical columns to impute.
        strategy : str
            Imputation strategy name.

        Returns
        -------
        tuple of (pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict)
            The three imputed DataFrames and a dict logging the imputation details.
        """
        train_df = train_df.copy()
        val_df = val_df.copy()
        test_df = test_df.copy()
        imputation_info = {}

        for col in categorical_cols:
            if col not in train_df.columns:
                continue

            # Check all splits for missing values (not just train)
            has_any_missing = (
                train_df[col].isna().any() or
                (len(val_df) > 0 and val_df[col].isna().any()) or
                test_df[col].isna().any()
            )
            if not has_any_missing:
                continue

            if strategy == 'mode':
                # Most frequent category in TRAINING set only
                mode_val = train_df[col].mode()
                if len(mode_val) > 0:
                    fill_value = mode_val[0]
                    imputation_info[col] = {'strategy': 'mode', 'value': str(fill_value)}
                else:
                    # Edge case: column is entirely NaN in training → use 'missing'
                    fill_value = 'missing'
                    imputation_info[col] = {'strategy': 'mode_fallback', 'value': 'missing'}
            elif strategy == 'constant':
                # Create an explicit 'missing' category — this can be useful
                # because the model can learn that "missing" is informative
                fill_value = 'missing'
                imputation_info[col] = {'strategy': 'constant', 'value': 'missing'}
            elif strategy == 'forward_fill':
                # Carry last valid category forward WITHIN each player (prevents
                # cross-player contamination when rows are interleaved by date).
                # If still NaN after ffill (start of a player's series), fill with 'missing'.
                train_df[col] = train_df.groupby('Player ID')[col].ffill().fillna('missing')
                val_df[col] = val_df.groupby('Player ID')[col].ffill().fillna('missing')
                test_df[col] = test_df.groupby('Player ID')[col].ffill().fillna('missing')
                imputation_info[col] = {'strategy': 'forward_fill', 'value': None}
                continue
            elif strategy == 'drop':
                imputation_info[col] = {'strategy': 'drop', 'value': None}
                continue
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            # Apply the SAME fill value to all three sets
            train_df[col] = train_df[col].fillna(fill_value)
            val_df[col] = val_df[col].fillna(fill_value)
            test_df[col] = test_df[col].fillna(fill_value)

        # If strategy is 'drop', remove rows with remaining NaN values
        if strategy == 'drop':
            cols_to_check = [c for c in categorical_cols if c in train_df.columns]
            train_df = train_df.dropna(subset=cols_to_check)
            val_df = val_df.dropna(subset=cols_to_check)
            test_df = test_df.dropna(subset=cols_to_check)

        return train_df, val_df, test_df, imputation_info

    def _encode_categorical_label(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        categorical_cols: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
        """
        Label encode categorical features: map categories → integers.

        === FIT on TRAIN, TRANSFORM on ALL ===

        The category → integer mapping is learned from training data only.
        Categories that appear in val/test but NOT in training are mapped
        to -1 (unseen category sentinel value).

        Example for 'Position' column:
            Training unique values: ['CD', 'ST', 'CAM', 'FB']
            Mapping: CD→0, ST→1, CAM→2, FB→3
            If test has 'GK' (not in training): GK → -1

        Label encoding is preferred for tree-based models (XGBoost, CatBoost)
        because they handle ordinal values efficiently. For linear models,
        one-hot encoding is usually better (see _encode_categorical_onehot).

        Parameters
        ----------
        train_df, val_df, test_df : pd.DataFrame
            The three data splits.
        categorical_cols : List[str]
            Categorical columns to encode.

        Returns
        -------
        tuple of (pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict)
            The three encoded DataFrames and a dict with the mappings used.
        """
        train_df = train_df.copy()
        val_df = val_df.copy()
        test_df = test_df.copy()
        encoding_info = {}

        for col in categorical_cols:
            if col not in train_df.columns:
                continue

            # Learn the mapping from TRAINING data only
            unique_vals = train_df[col].dropna().unique()
            mapping = {val: idx for idx, val in enumerate(unique_vals)}
            encoding_info[col] = {'type': 'label', 'mapping': mapping}

            # Apply the same mapping to all three sets
            train_df[col] = train_df[col].map(mapping)
            val_df[col] = val_df[col].map(mapping)
            test_df[col] = test_df[col].map(mapping)

            # Categories NOT seen in training will produce NaN after .map()
            # Replace these with -1 as a sentinel for "unseen category"
            train_df[col] = train_df[col].fillna(-1).astype(int)
            val_df[col] = val_df[col].fillna(-1).astype(int)
            test_df[col] = test_df[col].fillna(-1).astype(int)

        return train_df, val_df, test_df, encoding_info

    def _encode_categorical_onehot(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        categorical_cols: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict, List[str]]:
        """
        One-hot encode categorical features: create binary dummy columns.

        === FIT on TRAIN, TRANSFORM on ALL ===

        The encoder learns which categories exist from training data only.
        Categories that appear in val/test but NOT in training produce all-zero
        rows (via handle_unknown='ignore'), which is a safe default.

        Example for 'Position' with training categories ['CD', 'ST', 'CAM']:
            CD  → [1, 0, 0]  (Position_CD=1, Position_ST=0, Position_CAM=0)
            ST  → [0, 1, 0]
            CAM → [0, 0, 1]
            GK  → [0, 0, 0]  (unseen in training → all zeros)

        The original categorical column is DROPPED and replaced by the
        binary dummy columns. This increases dimensionality but avoids
        the implicit ordinality issue of label encoding.

        One-hot encoding is preferred for linear models (LogReg) where
        label encoding would incorrectly imply an ordered relationship
        between categories.

        Parameters
        ----------
        train_df, val_df, test_df : pd.DataFrame
            The three data splits.
        categorical_cols : List[str]
            Categorical columns to encode.

        Returns
        -------
        tuple of (pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict, List[str])
            - The three encoded DataFrames (original columns replaced by dummies)
            - Dict with encoding info (categories, dummy column names)
            - List of all new dummy column names created
        """
        encoding_info = {}
        all_dummy_cols = []

        for col in categorical_cols:
            if col not in train_df.columns:
                continue

            # FIT encoder on TRAINING data only
            # sparse_output=False: returns dense array (easier to work with)
            # handle_unknown='ignore': unseen categories in val/test → all zeros
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            train_encoded = encoder.fit_transform(train_df[[col]])

            # Create descriptive column names: 'Position_CD', 'Position_ST', etc.
            categories = encoder.categories_[0]
            dummy_cols = [f"{col}_{cat}" for cat in categories]

            # TRANSFORM all three sets using the SAME encoder
            train_dummies = pd.DataFrame(train_encoded, columns=dummy_cols, index=train_df.index)
            val_encoded = encoder.transform(val_df[[col]])
            val_dummies = pd.DataFrame(val_encoded, columns=dummy_cols, index=val_df.index)
            test_encoded = encoder.transform(test_df[[col]])
            test_dummies = pd.DataFrame(test_encoded, columns=dummy_cols, index=test_df.index)

            # Replace the original categorical column with the binary dummies.
            # Use .join() instead of concat+reset_index — this pins the dummy
            # columns to the correct rows via index matching, which is robust
            # even when indices are non-sequential (e.g., after dropna imputation).
            train_df = train_df.drop(columns=[col]).join(train_dummies)
            val_df = val_df.drop(columns=[col]).join(val_dummies)
            test_df = test_df.drop(columns=[col]).join(test_dummies)

            encoding_info[col] = {'type': 'one-hot', 'categories': list(categories), 'dummy_columns': dummy_cols}
            all_dummy_cols.extend(dummy_cols)

        return train_df, val_df, test_df, encoding_info, all_dummy_cols

    def _standardize_features(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        numerical_cols: List[str],
        scaler: Optional[StandardScaler] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Z-score standardize numerical features: (x - mean) / std.

        === FIT on TRAIN, TRANSFORM on ALL ===

        The mean and standard deviation are computed from TRAINING data only.
        The same values are applied to val/test. This means val/test values
        may fall outside [-3, 3] if their distribution differs from training
        — which is expected and correct behavior.

        Standardization is important for:
        - Linear models (LogReg): ensures all features are on the same scale,
          making coefficients comparable and regularization fair.
        - Other models that expect standardized inputs (e.g., neural networks).

        Standardization is NOT needed for:
        - Tree models (XGBoost, CatBoost): trees split on thresholds and
          are invariant to monotonic transformations of features.

        Only numerical features are standardized (not one-hot encoded dummies,
        which are already binary 0/1).

        Parameters
        ----------
        train_df, val_df, test_df : pd.DataFrame
            The three data splits.
        numerical_cols : List[str]
            Numerical columns to standardize.
        scaler : StandardScaler, optional
            Scaler to use. If None, uses self.scaler. Allows passing a
            separate scaler for treatment variables.

        Returns
        -------
        tuple of (pd.DataFrame, pd.DataFrame, pd.DataFrame)
            The three standardized DataFrames.
        """
        # Only standardize columns that actually exist in the data
        num_cols_present = [col for col in numerical_cols if col in train_df.columns]

        if not num_cols_present:
            return train_df, val_df, test_df

        if scaler is None:
            scaler = self.scaler

        # FIT: learn mean and std from training data only
        scaler.fit(train_df[num_cols_present])

        # TRANSFORM: apply the same mean and std to all three sets
        train_df = train_df.copy()
        train_df[num_cols_present] = scaler.transform(train_df[num_cols_present])

        if len(val_df) > 0:
            val_df = val_df.copy()
            val_df[num_cols_present] = scaler.transform(val_df[num_cols_present])

        test_df = test_df.copy()
        test_df[num_cols_present] = scaler.transform(test_df[num_cols_present])

        return train_df, val_df, test_df

    def create_dataset(
        self,
        target_variable: str,
        predictory_columns: List[str],
        lag: int = 0,
        include_previous_target: bool = False,
        target_horizon: int = 0,
        test_size: float = 0.2,
        val_size: float = 0.1,
        categorical_encoding: str = 'one-hot',
        missing_numeric: str = 'mean',
        missing_categorical: str = 'mode',
        standardize: bool = False,
        # === Treatment variable support (NEW) ===
        treatment_columns: Optional[List[str]] = None,
        treatment_horizon: Union[int, Dict[str, int]] = 0,
        treatment_lag: int = 0,
        standardize_treatment: Optional[bool] = None,
    ) -> Dict:
        """
        Create train/val/test datasets with proper anti-leakage preprocessing.

        This is the MAIN PUBLIC API of the data loader. It orchestrates the
        entire pipeline from loading processed data to returning ML-ready
        feature matrices (X), treatment matrices (T), and target vectors (y)
        for each split.

        === PIPELINE ORDER (critical for preventing data leakage) ===

        1.  Load data and merge rare activity categories
        2.  Validate that requested columns exist (covariates + treatment)
        3.  Classify covariate features as categorical vs numerical
        3b. Classify treatment features as categorical vs numerical
        4.  Create player-grouped lag features for covariates (BEFORE splitting)
        4b. Shift treatment columns by treatment_horizon (BEFORE splitting)
        4c. Create treatment lag features if treatment_lag > 0
        5.  Identify and classify the new lag columns
        5b. Shift target by target_horizon
        5c. Drop rows with NaN in treatment or target (from shifting)
        6.  *** SPLIT THE DATA (temporal date-block split) ***
        7.  Impute missing covariate values (fit on train only)
        7b. Impute missing treatment values (fit on train only)
        8.  Encode categorical covariates (fit on train only)
        8b. Encode categorical treatment (fit on train only)
        9.  Standardize numerical covariates (fit on train only)
        9b. Standardize numerical treatment if requested (fit on train only)
        10. Extract metadata (Date, Player ID) for error analysis
        11. Separate covariates (X), treatment (T), and target (y)

        CRITICAL: All fitting (imputation stats, encoding mappings, scaler
        parameters) is done on training data ONLY, then applied to
        validation and test sets. This is the core anti-leakage guarantee.

        Parameters
        ----------
        target_variable : str
            Column name to predict (e.g., 'Status Decrease').
        predictory_columns : List[str]
            Column names to use as covariates (X). Must not overlap with
            treatment_columns if treatment is specified.
        lag : int, default=0
            Number of lagged time steps to create for covariates. lag=3 creates
            t-1, t-2, t-3 for each predictor. Higher lag = more features but
            more NaN rows.
        include_previous_target : bool, default=False
            Whether to include lagged target as a feature (e.g., 'Status Decrease_t-1').
        target_horizon : int, default=0
            Number of time steps to shift the target variable forward within
            each player group. When target_horizon=1, the target at row t
            becomes the original target from row t+1 (next day).

            Use target_horizon=0 when:
                The outcome at row t reflects the result of yesterday's
                treatment (e.g., Status Decrease observed this morning).

            Use target_horizon=1 when:
                The outcome should reflect the effect of TODAY's treatment
                (decided at row t), which manifests TOMORROW (row t+1).
        test_size : float, default=0.2
            Proportion of dates for the test set.
        val_size : float, default=0.1
            Proportion of dates for the validation set.
        categorical_encoding : str, default='one-hot'
            'one-hot' (binary dummies) or 'label' (integer mapping).
        missing_numeric : str, default='mean'
            Strategy for numerical imputation: 'mean', 'median', 'zero',
            'forward_fill', 'drop'.
        missing_categorical : str, default='mode'
            Strategy for categorical imputation: 'mode', 'constant',
            'forward_fill', 'drop'.
        standardize : bool, default=False
            Whether to z-score standardize numerical covariates.
        treatment_columns : List[str] or None, default=None
            Column names to separate as treatment variables (T). When None
            or [], no treatment is extracted and the return dictionary only
            contains X and y. When provided, the specified columns are
            processed separately from covariates and returned as T_train,
            T_val, T_test. Any column in the dataset can be designated as
            a treatment variable — the choice depends on the experiment.
        treatment_horizon : int or Dict[str, int], default=0
            Row offset for pulling treatment values relative to the covariate
            row. Controls WHICH ROW the treatment comes from.

            If int: applies the same offset to ALL treatment columns.
                0 = treatment from the same row as covariates (default)
                1 = treatment from the next row

            If dict: maps individual column names to their offsets.
                Columns not in the dict default to 0. This supports mixed
                horizons where different treatment columns come from
                different time points.

            Example:
                treatment_horizon={
                    'Activity Type Today': 0,
                    'Total Distance % Yesterday': 1,
                }
        treatment_lag : int, default=0
            Number of lagged treatment features to create (for modeling
            sequential treatment effects). Usually 0 for single-decision
            causal analysis.
        standardize_treatment : bool or None, default=None
            Whether to z-score standardize numerical treatment variables.
            None: inherits the value of `standardize`.
            True/False: explicit override. Useful when you want standardized
            covariates but raw-scale treatment (common in causal methods).

        Returns
        -------
        Dict
            Dictionary containing:

            Always present:
            - X_train, y_train, meta_train: Training data
            - X_val, y_val, meta_val: Validation data (empty if val_size=0)
            - X_test, y_test, meta_test: Test data
            - feature_names: List of all covariate column names (after encoding)
            - categorical_features, numerical_features: Feature type lists
            - encoding_info: Details of categorical encoding applied
            - imputation_info: Details of missing value imputation applied
            - metadata: Dict of all configuration parameters and dataset sizes

            Present only when treatment_columns is provided:
            - T_train, T_val, T_test: Treatment variable matrices
            - treatment_feature_names: Treatment column names (after encoding)
            - treatment_categorical: Categorical treatment feature names
            - treatment_numerical: Numerical treatment feature names
            - treatment_encoding_info: Treatment encoding details
            - treatment_imputation_info: Treatment imputation details
        """
        # Normalize treatment_columns: None and [] both mean "no treatment"
        has_treatment = treatment_columns is not None and len(treatment_columns) > 0

        # =====================================================================
        # Step 1: Load processed data and reduce rare activity categories
        # =====================================================================
        df = self._verify_and_load_data()
        df = self._merge_activity_categories(df)

        # =====================================================================
        # Step 2: Validate that all requested columns exist in the data
        # =====================================================================
        if target_variable not in df.columns:
            raise ValueError(f"Target variable '{target_variable}' not found")

        missing_cols = [col for col in predictory_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Predictor columns not found: {missing_cols}")

        # Guard against the most common leakage mistake: including the target
        # variable as a predictor, which gives "perfect" but useless accuracy
        if target_variable in predictory_columns:
            raise ValueError(
                f"Target variable '{target_variable}' is included in predictory_columns. "
                f"This would cause data leakage (the model sees the answer as a feature). "
                f"Remove it from predictory_columns."
            )

        # Validate treatment columns exist
        if has_treatment:
            missing_treat = [col for col in treatment_columns if col not in df.columns]
            if missing_treat:
                raise ValueError(f"Treatment columns not found: {missing_treat}")

            # Ensure no overlap between predictors and treatment
            overlap = set(treatment_columns) & set(predictory_columns)
            if overlap:
                raise ValueError(
                    f"Columns cannot be both predictors and treatment: {overlap}. "
                    f"Treatment columns should NOT appear in predictory_columns."
                )

        # =====================================================================
        # Step 3: Classify covariate features as categorical or numerical
        # This determines how each feature will be imputed and encoded
        # =====================================================================
        categorical_features, numerical_features = self._identify_feature_types(
            predictory_columns, df, context='predictor'
        )

        # =====================================================================
        # Step 3b: Classify treatment features as categorical or numerical
        # Uses context='treatment' to suppress the Activity Type Today warning
        # (the user explicitly chose it as treatment, so no need to warn)
        # =====================================================================
        treat_categorical = []
        treat_numerical = []
        if has_treatment:
            treat_categorical, treat_numerical = self._identify_feature_types(
                treatment_columns, df, context='treatment'
            )

        # =====================================================================
        # Step 4: Create lag features for covariates (player-grouped, before split)
        # Must happen before splitting because lag features need the full
        # time series per player to compute correctly
        # =====================================================================
        df_lagged = self._create_lag_features(
            df=df,
            columns=predictory_columns,
            lag=lag,
            include_target=include_previous_target,
            target_variable=target_variable
        )

        # =====================================================================
        # Step 4b: Shift treatment columns by treatment_horizon
        # This aligns treatment data from other rows to the current covariate
        # row. E.g., treatment_horizon=1 pulls a value from the next row
        # back to the current row. Must happen BEFORE splitting.
        # =====================================================================
        if has_treatment:
            df_lagged = self._apply_treatment_horizon(
                df_lagged, treatment_columns, treatment_horizon
            )

        # =====================================================================
        # Step 4c: Create treatment lag features if treatment_lag > 0
        # For modeling sequential treatment effects (e.g., how the last 3 days
        # of training intensity affect the outcome).
        # =====================================================================
        treatment_lag_columns = []
        if has_treatment and treatment_lag > 0:
            df_lagged, treatment_lag_columns = self._create_treatment_lag_features(
                df_lagged, treatment_columns, treatment_lag
            )

        # =====================================================================
        # Step 5: Build the list of lag column names and classify them
        # Lagged features inherit the type of their parent column:
        # e.g., 'Position_t-1' is categorical because 'Position' is categorical
        # =====================================================================
        lag_columns = []
        if lag > 0:
            for col in predictory_columns:
                for t in range(1, lag + 1):
                    lag_columns.append(f"{col}_t-{t}")
            if include_previous_target and lag >= 1:
                for t in range(1, lag + 1):
                    lag_columns.append(f"{target_variable}_t-{t}")

        # Classify lagged features based on their parent column's type
        lagged_categorical = []
        lagged_numerical = []
        for lag_col in lag_columns:
            # Extract original column name by removing the '_t-N' suffix.
            # Use rsplit to split from the RIGHT at '_t-' to correctly handle
            # column names that contain underscores (e.g., 'Sleep Quality (z)_t-1').
            orig_col = lag_col.rsplit('_t-', 1)[0] if '_t-' in lag_col else lag_col
            if orig_col in categorical_features:
                lagged_categorical.append(lag_col)
            else:
                lagged_numerical.append(lag_col)

        # Combine original + lagged covariate features
        all_categorical = categorical_features + lagged_categorical
        all_numerical = numerical_features + lagged_numerical

        # Classify treatment lag columns based on their parent treatment type
        treat_lagged_categorical = []
        treat_lagged_numerical = []
        for lag_col in treatment_lag_columns:
            orig_col = lag_col.rsplit('_t-', 1)[0] if '_t-' in lag_col else lag_col
            if orig_col in treat_categorical:
                treat_lagged_categorical.append(lag_col)
            else:
                treat_lagged_numerical.append(lag_col)

        # Combine original + lagged treatment features
        all_treat_categorical = treat_categorical + treat_lagged_categorical
        all_treat_numerical = treat_numerical + treat_lagged_numerical

        # =====================================================================
        # Step 5b: Shift target variable forward by target_horizon steps
        # This makes the model predict the FUTURE target given current features.
        # E.g., with target_horizon=1: features from day t predict target from day t+1.
        # The shift is done WITHIN each player group (shift(-N) pulls future values back).
        # =====================================================================
        if target_horizon > 0:
            df_lagged = df_lagged.sort_values(['Player ID', 'Date']).copy()
            df_lagged[target_variable] = df_lagged.groupby('Player ID')[target_variable].shift(-target_horizon)

        # =====================================================================
        # Step 5c: Drop rows where target or treatment is NaN after shifting
        # This handles both target_horizon shifts and treatment_horizon shifts.
        # Rows at the end of each player's series will have NaN after
        # shift(-N) because there's no future data to pull from.
        # =====================================================================
        drop_cols = [target_variable]
        if has_treatment:
            drop_cols.extend(treatment_columns)
        df_lagged = df_lagged.dropna(subset=drop_cols).copy()

        # After shifting, a binary target (0/1) becomes float due to NaN introduction.
        # Restore integer type ONLY if the values are genuinely integer-like (e.g., 0.0/1.0).
        # This preserves continuous targets (e.g., match-day performance) for causal DTR.
        if target_horizon > 0 and df_lagged[target_variable].dtype == float:
            vals = df_lagged[target_variable].dropna()
            if len(vals) > 0 and (vals == vals.astype(int)).all():
                df_lagged[target_variable] = df_lagged[target_variable].astype(int)

        # =====================================================================
        # Step 6: *** SPLIT FIRST *** — THE MOST CRITICAL STEP
        # Everything AFTER this point is fit on train only, then transformed
        # on all sets. This prevents data leakage from test → train.
        # =====================================================================
        train_df, val_df, test_df = self._temporal_split(df_lagged, test_size, val_size)

        # =====================================================================
        # Step 7: Impute missing covariate data (FIT on train, TRANSFORM on all)
        # =====================================================================
        train_df, val_df, test_df, num_imputation_info = self._impute_numerical(
            train_df, val_df, test_df, all_numerical, missing_numeric
        )
        train_df, val_df, test_df, cat_imputation_info = self._impute_categorical(
            train_df, val_df, test_df, all_categorical, missing_categorical
        )
        imputation_info = {'numerical': num_imputation_info, 'categorical': cat_imputation_info}

        # =====================================================================
        # Step 7b: Impute missing treatment data (FIT on train, TRANSFORM on all)
        # Treatment imputation is tracked separately from covariate imputation.
        # =====================================================================
        treat_imputation_info = {}
        if has_treatment:
            train_df, val_df, test_df, treat_num_imp = self._impute_numerical(
                train_df, val_df, test_df, all_treat_numerical, missing_numeric
            )
            train_df, val_df, test_df, treat_cat_imp = self._impute_categorical(
                train_df, val_df, test_df, all_treat_categorical, missing_categorical
            )
            treat_imputation_info = {'numerical': treat_num_imp, 'categorical': treat_cat_imp}

        # =====================================================================
        # Step 8: Encode categorical covariates (FIT on train, TRANSFORM on all)
        # =====================================================================
        if categorical_encoding == 'one-hot':
            # One-hot: replace each categorical column with binary dummy columns
            train_df, val_df, test_df, encoding_info, encoded_categorical = self._encode_categorical_onehot(
                train_df, val_df, test_df, all_categorical
            )
            all_features = all_numerical + encoded_categorical
        else:  # label encoding
            # Label: replace category strings with integer codes
            train_df, val_df, test_df, encoding_info = self._encode_categorical_label(
                train_df, val_df, test_df, all_categorical
            )
            encoded_categorical = all_categorical
            all_features = all_numerical + all_categorical

        # =====================================================================
        # Step 8b: Encode categorical treatment (FIT on train, TRANSFORM on all)
        # Uses the SAME encoding scheme as covariates for consistency.
        # =====================================================================
        treat_encoding_info = {}
        treat_encoded_categorical = []
        all_treat_features = []
        if has_treatment:
            if categorical_encoding == 'one-hot':
                train_df, val_df, test_df, treat_encoding_info, treat_encoded_categorical = \
                    self._encode_categorical_onehot(
                        train_df, val_df, test_df, all_treat_categorical
                    )
                all_treat_features = all_treat_numerical + treat_encoded_categorical
            else:
                train_df, val_df, test_df, treat_encoding_info = self._encode_categorical_label(
                    train_df, val_df, test_df, all_treat_categorical
                )
                treat_encoded_categorical = all_treat_categorical
                all_treat_features = all_treat_numerical + all_treat_categorical

        # =====================================================================
        # Step 9: Standardize covariates if requested (FIT on train, TRANSFORM on all)
        # Only applied to numerical features (not one-hot dummies)
        # =====================================================================
        if standardize:
            train_df, val_df, test_df = self._standardize_features(
                train_df, val_df, test_df, all_numerical, scaler=self.scaler
            )

        # =====================================================================
        # Step 9b: Standardize treatment if requested (FIT on train, TRANSFORM on all)
        # Uses a SEPARATE scaler so treatment and covariate standardization
        # are fully independent. Default inherits from `standardize`.
        # =====================================================================
        effective_std_treat = standardize_treatment if standardize_treatment is not None else standardize
        if has_treatment and effective_std_treat and all_treat_numerical:
            train_df, val_df, test_df = self._standardize_features(
                train_df, val_df, test_df, all_treat_numerical,
                scaler=self.treatment_scaler
            )

        # =====================================================================
        # Step 10: Extract metadata columns for error analysis
        # These are NOT features — they identify WHO (player) and WHEN (date)
        # each prediction was made for, enabling post-hoc error analysis
        # =====================================================================
        meta_cols = ['Date', 'Player ID', 'Playerkey']
        available_meta_cols = [col for col in meta_cols if col in train_df.columns]

        train_meta = train_df[available_meta_cols].copy() if available_meta_cols else pd.DataFrame()
        val_meta = val_df[available_meta_cols].copy() if len(val_df) > 0 and available_meta_cols else pd.DataFrame()
        test_meta = test_df[available_meta_cols].copy() if available_meta_cols else pd.DataFrame()

        # =====================================================================
        # Step 11: Separate covariates (X), treatment (T), and target (y)
        # =====================================================================
        X_train = train_df[all_features].copy()
        y_train = train_df[target_variable].copy()

        X_val = val_df[all_features].copy() if len(val_df) > 0 else pd.DataFrame()
        y_val = val_df[target_variable].copy() if len(val_df) > 0 else pd.Series(dtype=float)

        X_test = test_df[all_features].copy()
        y_test = test_df[target_variable].copy()

        # Store feature types as instance variables for potential reuse
        self.categorical_features = encoded_categorical
        self.numerical_features = all_numerical

        # Build the result dictionary
        result = {
            'X_train': X_train,           # Training covariate matrix
            'y_train': y_train,           # Training target vector
            'meta_train': train_meta,     # Training metadata (Date, Player ID)
            'X_val': X_val,               # Validation covariate matrix
            'y_val': y_val,               # Validation target vector
            'meta_val': val_meta,         # Validation metadata
            'X_test': X_test,             # Test covariate matrix
            'y_test': y_test,             # Test target vector
            'meta_test': test_meta,       # Test metadata
            'feature_names': all_features,          # All covariate column names
            'categorical_features': self.categorical_features,  # Categorical feature names
            'numerical_features': self.numerical_features,      # Numerical feature names
            'encoding_info': encoding_info,         # How categoricals were encoded
            'imputation_info': imputation_info,     # How missing values were filled
            'metadata': {                           # Configuration and dataset sizes
                'target_variable': target_variable,
                'original_predictors': predictory_columns,
                'lag': lag,
                'include_previous_target': include_previous_target,
                'target_horizon': target_horizon,
                'test_size': test_size,
                'val_size': val_size,
                'categorical_encoding': categorical_encoding,
                'missing_numeric': missing_numeric,
                'missing_categorical': missing_categorical,
                'standardize': standardize,
                'treatment_columns': treatment_columns if treatment_columns else [],
                'treatment_horizon': treatment_horizon,
                'treatment_lag': treatment_lag,
                'standardize_treatment': effective_std_treat if has_treatment else None,
                'n_train': len(X_train),
                'n_val': len(X_val),
                'n_test': len(X_test)
            }
        }

        # Add treatment data if treatment columns were specified
        if has_treatment:
            T_train = train_df[all_treat_features].copy()
            T_val = val_df[all_treat_features].copy() if len(val_df) > 0 else pd.DataFrame()
            T_test = test_df[all_treat_features].copy()

            result.update({
                'T_train': T_train,                           # Training treatment matrix
                'T_val': T_val,                               # Validation treatment matrix
                'T_test': T_test,                             # Test treatment matrix
                'treatment_feature_names': all_treat_features,  # Treatment column names (after encoding)
                'treatment_categorical': treat_encoded_categorical,  # Categorical treatment names
                'treatment_numerical': all_treat_numerical,         # Numerical treatment names
                'treatment_encoding_info': treat_encoding_info,     # Treatment encoding details
                'treatment_imputation_info': treat_imputation_info, # Treatment imputation details
            })

        return result
