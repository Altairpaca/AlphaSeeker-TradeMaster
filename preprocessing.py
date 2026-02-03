import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class FinancialDataPreprocessor:
    """
    Financial time series data preprocessor
    Handles feature missing values and creates new features
    """

    def __init__(self, save_path='dataset/train_processed.csv'):
        self.save_path = save_path
        self.train_stats = {}  # Store training set statistics for test set
        self.feature_cols = [f'feature_{i}' for i in range(1, 31)]

    def fit(self, train_df):
        """
        Fit on training set, calculate necessary statistics
        """
        # self.train_df = pd.DataFrame(np.where(np.isinf(train_df), np.nan, train_df), columns=train_df.columns)
        self.train_df = train_df.copy()
        self._compute_train_stats()
        return self

    def transform(self, df, is_train=False):
        """
        Transform data (training or test set)
        is_train: Whether it's training set (determines if new statistics are calculated)
        """
        df_processed = df.copy()

        # 1. Handle missing values at the start
        df_processed = self._handle_start_missing(df_processed, is_train)

        # 2. Special handling for feature_11
        df_processed = self._handle_feature_11_special(df_processed, is_train)

        # 3. Handle other missing values in the middle
        df_processed = self._handle_general_missing(df_processed, is_train)

        # 4. Feature engineering
        df_processed = self._feature_engineering(df_processed, is_train)

        # 5. Final check
        df_processed = self._final_check(df_processed)

        if is_train:
            # Save processed training set
            Path('dataset').mkdir(exist_ok=True)
            df_processed.to_csv(self.save_path, index=False)
            print(f"✅ Processed training set saved to: {self.save_path}")
            print(f"📊 Data shape: {df_processed.shape}")
            print(
                f"🔍 Remaining missing values: {df_processed[self.feature_cols].isnull().sum().sum()}")

        return df_processed

    def fit_transform(self, train_df):
        """Fit and transform training set"""
        return self.fit(train_df).transform(train_df, is_train=True)

    def _compute_train_stats(self):
        """Calculate statistics for training set"""
        # 1. Find the first valid value for each feature (could be at any minute)
        self.start_fill_values = {}
        # Track where the first valid value is for each feature
        self.first_valid_positions = {}

        for feature in self.feature_cols:
            # Find the first non-null value for this feature in the entire training set
            # Sort by date_id and minute_id to ensure chronological order
            sorted_df = self.train_df.sort_values(['date_id', 'minute_id'])

            # Find first non-null value
            first_valid_mask = sorted_df[feature].notna()
            if first_valid_mask.any():
                first_valid_row = sorted_df[first_valid_mask].iloc[0]
                self.start_fill_values[feature] = first_valid_row[feature]
                self.first_valid_positions[feature] = {
                    'date_id': first_valid_row['date_id'],
                    'minute_id': first_valid_row['minute_id']
                }
            else:
                # If all values are missing for this feature (shouldn't happen)
                self.start_fill_values[feature] = 0
                self.first_valid_positions[feature] = {
                    'date_id': 0, 'minute_id': 0}

            print(f"{feature}: First valid value = {self.start_fill_values[feature]} at "
                  f"date_id={self.first_valid_positions[feature]['date_id']}, "
                  f"minute_id={self.first_valid_positions[feature]['minute_id']}")

        # 2. Global statistics for each feature
        self.global_stats = {}
        for feature in self.feature_cols:
            self.global_stats[feature] = {
                'mean': self.train_df[feature].mean(),
                'median': self.train_df[feature].median(),
                'std': self.train_df[feature].std(),
                'q25': self.train_df[feature].quantile(0.25),
                'q75': self.train_df[feature].quantile(0.75)
            }

        # 3. Special statistics for feature_11
        self.feature_11_stats = {
            'day_236_means': {},  # Values at minute 236 for each day
            'day_means': {}       # Daily means
        }

        # Calculate values at minute 236 for each day
        for date_id in self.train_df['date_id'].unique():
            day_data = self.train_df[self.train_df['date_id'] == date_id]

            # Value at minute 236
            minute_236 = day_data[day_data['minute_id'] == 236]['feature_11']
            if len(minute_236) > 0 and not pd.isna(minute_236.values[0]):
                self.feature_11_stats['day_236_means'][date_id] = minute_236.values[0]

            # Daily mean
            day_mean = day_data['feature_11'].mean()
            self.feature_11_stats['day_means'][date_id] = day_mean

    def _handle_start_missing(self, df, is_train):
        """Handle missing values at the start of the dataset"""
        df_processed = df.copy()

        for feature in self.feature_cols:
            # Get the first valid position for this feature
            first_valid_date = self.first_valid_positions[feature]['date_id']
            first_valid_minute = self.first_valid_positions[feature]['minute_id']

            # Mark all records that are chronologically before the first valid value
            # and have missing values for this feature
            start_missing_mask = (
                # Records that are chronologically before the first valid value
                ((df_processed['date_id'] < first_valid_date) |
                 ((df_processed['date_id'] == first_valid_date) &
                 (df_processed['minute_id'] < first_valid_minute))) &
                # And have missing values for this feature
                df_processed[feature].isnull()
            )

            if start_missing_mask.any():
                fill_value = self.start_fill_values[feature]
                df_processed.loc[start_missing_mask, feature] = fill_value

                # Create missing indicator
                df_processed[f'{feature}_missing_start'] = start_missing_mask.astype(
                    int)

                if is_train and start_missing_mask.sum() > 0:
                    print(
                        f"{feature}: Filled {start_missing_mask.sum()} start missing values with {fill_value}")
            else:
                df_processed[f'{feature}_missing_start'] = 0


        return df_processed

    def _handle_feature_11_special(self, df, is_train):
        """Special handling for missing values in feature_11"""
        df_processed = df.copy()

        # Mark original missing positions
        original_missing = df_processed['feature_11'].isnull()

        # 1. Handle systematic missing at minutes 237-238
        systemic_mask = (
            original_missing &
            df_processed['minute_id'].isin([237, 238])
        )

        for idx in df_processed[systemic_mask].index:
            date_id = df_processed.at[idx, 'date_id']

            # Try to fill with value at minute 236 of the same day
            if date_id in self.feature_11_stats['day_236_means']:
                fill_value = self.feature_11_stats['day_236_means'][date_id]
            elif date_id in self.feature_11_stats['day_means']:
                # Use daily mean
                fill_value = self.feature_11_stats['day_means'][date_id]
            else:
                # Use global mean
                fill_value = self.global_stats['feature_11']['mean']

            df_processed.at[idx, 'feature_11'] = fill_value

        # First try to interpolate with values before and after within the same day
        for date_id in df_processed['date_id'].unique():
            date_mask = df_processed['date_id'] == date_id
            date_data = df_processed.loc[date_mask, 'feature_11']

            if date_data.isnull().any():
                # Linear interpolation within the day
                df_processed.loc[date_mask, 'feature_11'] = date_data.interpolate(
                    method='linear', limit_direction='both'
                )

        # 3. Create indicator features
        df_processed['feature_11_missing_systemic'] = systemic_mask.astype(int)
        df_processed['feature_11_missing_other'] = (
            original_missing & ~systemic_mask
        ).astype(int)

        return df_processed

    def _handle_general_missing(self, df, is_train):
        """Handle general missing values in the middle"""
        df_processed = df.copy()

        for feature in self.feature_cols:
            if feature == 'feature_11':
                continue  # Already handled separately

            # Mark original missing
            original_missing = df_processed[feature].isnull()

            if original_missing.any():
                # 1. First forward fill within each day
                df_processed[feature] = df_processed.groupby('date_id')[feature].transform(
                    lambda x: x.fillna(method='ffill')
                )

                # 2. If still missing, backward fill
                still_missing = df_processed[feature].isnull()
                if still_missing.any():
                    df_processed[feature] = df_processed.groupby('date_id')[feature].transform(
                        lambda x: x.fillna(method='bfill')
                    )

                # 3. If still missing, use global statistics from training set
                still_missing = df_processed[feature].isnull()
                if still_missing.any():
                    fill_value = self.global_stats[feature]['mean']
                    df_processed.loc[still_missing, feature] = fill_value

                # Create missing indicator
                df_processed[f'{feature}_was_missing'] = original_missing.astype(
                    int)
            else:
                df_processed[f'{feature}_was_missing'] = 0
        return df_processed

    def _feature_engineering(self, df, is_train):
        """Create new features based on missing patterns and time"""
        df_engineered = df.copy()

        # 1. Time features
        # df_engineered['minute_of_day'] = df_engineered['minute_id']
        # df_engineered['hour_of_day'] = df_engineered['minute_id'] // 60
        # df_engineered['is_morning'] = (
        #     df_engineered['hour_of_day'] < 12).astype(int)
        # df_engineered['is_afternoon'] = ((df_engineered['hour_of_day'] >= 12) &
        #                                  (df_engineered['hour_of_day'] < 16)).astype(int)
        df_engineered['is_near_close'] = (
            df_engineered['minute_id'] >= 230).astype(int)

        # 2. Trading day progress
        df_engineered['trading_progress'] = df_engineered['minute_id'] / 239.0

        # 3. Missing value summary features
        missing_cols = [
            col for col in df_engineered.columns if '_missing' in col or '_was_missing' in col]
        if missing_cols:
            df_engineered['total_missing_features'] = df_engineered[missing_cols].sum(
                axis=1)

        # 4. Rolling statistics features (example)
        for feature in ['feature_11', 'feature_18', 'feature_21', 'feature_25']:
            if feature in df_engineered.columns:
                # Daily rolling mean
                df_engineered[f'{feature}_rolling_mean_10'] = df_engineered.groupby('date_id')[feature].transform(
                    lambda x: x.rolling(10, min_periods=1).mean()
                )

        # 5. Feature interactions (example)
        if 'feature_11' in df_engineered.columns and 'feature_18' in df_engineered.columns:
            df_engineered['f11_f18_interaction'] = df_engineered['feature_11'] * \
                df_engineered['feature_18']

        if 'feature_21' in df_engineered.columns and 'feature_25' in df_engineered.columns:
            df_engineered['f21_f25_interaction'] = df_engineered['feature_21'] * \
                df_engineered['feature_25']

        return df_engineered

    def _final_check(self, df):
        """Final check to ensure no missing values"""
        # Check all feature columns
        for feature in self.feature_cols:
            if df[feature].isnull().any():
                # Fill any remaining missing with median
                median_val = df[feature].median()
                df[feature] = df[feature].fillna(median_val)

        # Check new features
        for col in df.columns:
            if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                if df[col].isnull().any():
                    df[col] = df[col].fillna(df[col].median())

        return df

    def save_processor(self, path='dataset/preprocessor.pkl'):
        """Save preprocessor state"""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'start_fill_values': self.start_fill_values,
                'first_valid_positions': self.first_valid_positions,
                'global_stats': self.global_stats,
                'feature_11_stats': self.feature_11_stats
            }, f)
        print(f"💾 Preprocessor saved to: {path}")


# Main processing pipeline
def main():
    # 1. Load data
    print("📥 Loading data...")
    train_df = pd.read_csv('dataset/train_v2.csv')  # Assume file is in current directory
    train_df = pd.DataFrame(
        np.where(np.isinf(train_df), np.nan, train_df), columns=train_df.columns)
    test_df = pd.read_csv(
        'dataset/test_v2.csv') if Path('dataset/test_v2.csv').exists() else None

    # 2. Create and fit preprocessor
    print("🔄 Creating preprocessor...")
    preprocessor = FinancialDataPreprocessor(
        save_path='dataset/train_processed.csv')

    # 3. Process training set
    print("🔧 Processing training set...")
    train_processed = preprocessor.fit_transform(train_df)

    # 4. Save preprocessor state (for test set)
    preprocessor.save_processor('dataset/preprocessor.pkl')

    # 5. Process test set (if exists)
    if test_df is not None:
        print("🔧 Processing test set...")
        test_processed = preprocessor.transform(test_df, is_train=False)
        test_processed.to_csv('dataset/test_processed.csv', index=False)
        print(f"✅ Processed test set saved to: dataset/test_processed.csv")
        print(f"📊 Test set shape: {test_processed.shape}")

    # 6. Output summary
    print("\n" + "="*50)
    print("🎉 Preprocessing completed!")
    print("="*50)
    print(f"Original training set shape: {train_df.shape}")
    print(f"Processed training set shape: {train_processed.shape}")
    print(
        f"Number of new features: {train_processed.shape[1] - train_df.shape[1]}")

    # Check missing values
    feature_cols = [f'feature_{i}' for i in range(1, 31)]
    remaining_missing = train_processed[feature_cols].isnull().sum().sum()
    print(f"Remaining missing values in feature columns: {remaining_missing}")

    # Show new features
    new_cols = set(train_processed.columns) - set(train_df.columns)
    print(f"\nNew feature examples ({len(new_cols)} total):")
    for i, col in enumerate(sorted(list(new_cols))[:20]):  # Only show first 20
        print(f"  {i+1:2d}. {col}")
    if len(new_cols) > 20:
        print(f"  ... and {len(new_cols)-20} more features")

    return preprocessor, train_processed


if __name__ == "__main__":
    # Run preprocessing
    preprocessor, train_processed = main()

    # Optional: Verify processing results
    print("\n" + "="*50)
    print("🔍 Verifying processing results")
    print("="*50)

    # Check missing values for specific features
    special_features = ['feature_11', 'feature_18', 'feature_21', 'feature_25']
    for feat in special_features:
        if feat in train_processed.columns:
            missing = train_processed[feat].isnull().sum()
            print(f"{feat}: Remaining missing values = {missing}")

    # Check time continuity
    date_gaps = train_processed['date_id'].diff().value_counts().head()
    print(f"\nDate ID gap distribution:")
    print(date_gaps)

    # Save column list for later modeling
    feature_columns = [col for col in train_processed.columns
                       if not col.startswith('target') and col not in ['id', 'date_id', 'stock_id']]

    pd.Series(feature_columns).to_csv(
        'dataset/feature_columns.txt', index=False, header=False)
    print(f"\n📋 Feature list saved to: dataset/feature_columns.txt")
