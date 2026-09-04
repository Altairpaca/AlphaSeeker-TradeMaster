import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

print("Starting the XGBoost competition pipeline...")


# 1. LOAD DATA

print("Loading data...")
train_df = pd.read_csv('dataset/train_v2.csv')
test_df = pd.read_csv('dataset/test_v2.csv')

# Ensure chronological order
train_df = train_df.sort_values(['date_id', 'minute_id']).reset_index(drop=True)
test_df = test_df.sort_values(['date_id', 'minute_id']).reset_index(drop=True)

# Replace inf with nan
train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
test_df.replace([np.inf, -np.inf], np.nan, inplace=True)


# 2. FEATURE ENGINEERING

print("Engineering Z-scores and interactions...")
feature_cols = [f'feature_{i}' for i in range(1, 31)]

# 1. Full-day Z-score.
# This matches the offline competition feature path. It is NOT causal for a
# live minute-by-minute interpretation because later minutes of the same day
# contribute to the mean/std. See the repository README for the information-set audit.
for i in range(1, 31):
    col = f'feature_{i}'
    train_df[f'{col}_z'] = train_df.groupby('date_id')[col].transform(lambda x: (x - x.mean()) / (x.std() + 1e-5))
    test_df[f'{col}_z'] = test_df.groupby('date_id')[col].transform(lambda x: (x - x.mean()) / (x.std() + 1e-5))
    feature_cols.append(f'{col}_z')

# 2. Time features (240-minute competition day)
train_df['minute_sin'] = np.sin(2 * np.pi * train_df['minute_id'] / 240.0)
train_df['minute_cos'] = np.cos(2 * np.pi * train_df['minute_id'] / 240.0)
test_df['minute_sin'] = np.sin(2 * np.pi * test_df['minute_id'] / 240.0)
test_df['minute_cos'] = np.cos(2 * np.pi * test_df['minute_id'] / 240.0)
feature_cols.extend(['minute_sin', 'minute_cos'])

# 3. NaN indicators
for i in range(1, 31):
    col = f'feature_{i}'
    train_df[f'{col}_nan'] = train_df[col].isna().astype(int)
    test_df[f'{col}_nan'] = test_df[col].isna().astype(int)
    feature_cols.append(f'{col}_nan')


train_df['feat8z_x_feat27z'] = train_df['feature_8_z'] * train_df['feature_27_z']
test_df['feat8z_x_feat27z'] = test_df['feature_8_z'] * test_df['feature_27_z']

train_df['feat10z_x_feat6z'] = train_df['feature_10_z'] * train_df['feature_6_z']
test_df['feat10z_x_feat6z'] = test_df['feature_10_z'] * test_df['feature_6_z']

train_df['feat8z_minus_feat27z'] = train_df['feature_8_z'] - train_df['feature_27_z']
test_df['feat8z_minus_feat27z'] = test_df['feature_8_z'] - test_df['feature_27_z']

train_df['feat10z_minus_feat6z'] = train_df['feature_10_z'] - train_df['feature_6_z']
test_df['feat10z_minus_feat6z'] = test_df['feature_10_z'] - test_df['feature_6_z']

epsilon = 1e-5
train_df['feat8z_div_feat27z'] = train_df['feature_8_z'] / (train_df['feature_27_z'] + epsilon)
test_df['feat8z_div_feat27z'] = test_df['feature_8_z'] / (test_df['feature_27_z'] + epsilon)

train_df['feat10z_div_feat6z'] = train_df['feature_10_z'] / (train_df['feature_6_z'] + epsilon)
test_df['feat10z_div_feat6z'] = test_df['feature_10_z'] / (test_df['feature_6_z'] + epsilon)

feature_cols.extend([
    'feat8z_x_feat27z', 'feat10z_x_feat6z',
    'feat8z_minus_feat27z', 'feat10z_minus_feat6z',
    'feat8z_div_feat27z', 'feat10z_div_feat6z'
])

# 3. TIME-SERIES DATA SPLITTING

# Chronological holdout by date. This prevents random row mixing across the
# holdout boundary, but it does not make the full-day features above causal.
dates = train_df['date_id'].unique()
split_point = int(len(dates) * 0.85)
train_date_limit = dates[split_point]

X_train = train_df[train_df['date_id'] <= train_date_limit][feature_cols].values
X_val = train_df[train_df['date_id'] > train_date_limit][feature_cols].values

target_cols = ['target_short', 'target_medium', 'target_long']
y_train = train_df[train_df['date_id'] <= train_date_limit][target_cols].values
y_val = train_df[train_df['date_id'] > train_date_limit][target_cols].values

X_test = test_df[feature_cols].values

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")


# 4. TRAIN XGBOOST MODELS

print("\n" + "=" * 80)
print("Training XGBoost models (MAE objective)")
print("=" * 80)

xgb_models = {}
xgb_val_predictions = {}

TARGET_WEIGHTS = {'short': 0.1, 'medium': 0.3, 'long': 0.6}

for i, target_name in enumerate(['short', 'medium', 'long']):
    print(f"\n--- Training XGBoost for target_{target_name} ---")

    xgb_params = {
        'objective': 'reg:absoluteerror',
        'tree_method': 'hist',
        'max_depth': 5,
        'learning_rate': 0.02,
        'n_estimators': 1500,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42,
        'eval_metric': 'mae',
        'early_stopping_rounds': 50,
        'n_jobs': -1
    }

    model = xgb.XGBRegressor(**xgb_params)

    model.fit(
        X_train, y_train[:, i],
        eval_set=[(X_val, y_val[:, i])],
        verbose=100
    )

    val_pred = model.predict(X_val)

    # Competition-time post-processing: center predictions to reduce bias.
    val_pred -= val_pred.mean()

    val_mae = mean_absolute_error(y_val[:, i], val_pred)
    print(f"Validation MAE: {val_mae:.6f}")

    xgb_models[target_name] = model
    xgb_val_predictions[target_name] = val_pred

# Weighted MAE used by the competition objective.
xgb_weighted_mae = (
    TARGET_WEIGHTS['short'] * mean_absolute_error(y_val[:, 0], xgb_val_predictions['short']) +
    TARGET_WEIGHTS['medium'] * mean_absolute_error(y_val[:, 1], xgb_val_predictions['medium']) +
    TARGET_WEIGHTS['long'] * mean_absolute_error(y_val[:, 2], xgb_val_predictions['long'])
)

print("\n" + "=" * 80)
print("Competition evaluation summary")
print("=" * 80)
print(f"Short MAE:  {mean_absolute_error(y_val[:, 0], xgb_val_predictions['short']):.6f} (Weight: 0.1)")
print(f"Medium MAE: {mean_absolute_error(y_val[:, 1], xgb_val_predictions['medium']):.6f} (Weight: 0.3)")
print(f"Long MAE:   {mean_absolute_error(y_val[:, 2], xgb_val_predictions['long']):.6f} (Weight: 0.6)")
print(f"FINAL Weighted MAE: {xgb_weighted_mae:.6f}")
print("=" * 80)


# 5. PREDICT AND SAVE

print("\nGenerating final predictions...")
test_pred_xgb_short = xgb_models['short'].predict(X_test)
test_pred_xgb_medium = xgb_models['medium'].predict(X_test)
test_pred_xgb_long = xgb_models['long'].predict(X_test)

# Competition-time mean-centering.
test_pred_xgb_short -= test_pred_xgb_short.mean()
test_pred_xgb_medium -= test_pred_xgb_medium.mean()
test_pred_xgb_long -= test_pred_xgb_long.mean()

submission_xgb = pd.DataFrame({
    'id': test_df['id'],
    'target_short': test_pred_xgb_short,
    'target_medium': test_pred_xgb_medium,
    'target_long': test_pred_xgb_long
})

submission_xgb.to_csv('submission_xgb.csv', index=False)
print("Submission saved: submission_xgb.csv")