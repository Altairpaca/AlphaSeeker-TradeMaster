import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("🚀 Starting the Ultimate LightGBM Pipeline...")

# ==========================================
# 1. LOAD RAW DATA
# ==========================================
train_df = pd.read_csv('dataset/train_v2.csv')
test_df = pd.read_csv('dataset/test_v2.csv')

# Ensure chronological order (CRITICAL for Time Series)
train_df = train_df.sort_values(['date_id', 'minute_id']).reset_index(drop=True)
test_df = test_df.sort_values(['date_id', 'minute_id']).reset_index(drop=True)

# ==========================================
# 2. THE SENIOR'S HINT: "Even Number" Features
# ==========================================
def generate_features(df):
    df_feat = df.copy()
    
    # Original features
    base_features = [f'feature_{i}' for i in range(1, 31)]
    
    # Fill NAs with 0 (LightGBM actually handles NAs natively, but 0 is safe for diffs)
    df_feat[base_features] = df_feat[base_features].fillna(0)
    
    # 🌟 EVEN NUMBER WINDOWS (10m, 20m, 60m, 120m) 🌟
    # We only apply this to a few key features to prevent memory overload
    # Assume feature_11, 18, 21, 25 are the most important based on previous analysis
    key_features = ['feature_11', 'feature_18', 'feature_21', 'feature_25']
    
    for f in key_features:
        # Momentum (Price change over even periods)
        df_feat[f'{f}_diff_10'] = df_feat.groupby('date_id')[f].diff(10)
        df_feat[f'{f}_diff_20'] = df_feat.groupby('date_id')[f].diff(20)
        df_feat[f'{f}_diff_60'] = df_feat.groupby('date_id')[f].diff(60)
        df_feat[f'{f}_diff_120'] = df_feat.groupby('date_id')[f].diff(120)
        
        # Volatility (Standard deviation over even periods)
        df_feat[f'{f}_vol_20'] = df_feat.groupby('date_id')[f].transform(lambda x: x.rolling(20).std())
        df_feat[f'{f}_vol_60'] = df_feat.groupby('date_id')[f].transform(lambda x: x.rolling(60).std())

    # Time of day feature (Progress through the 240-minute day)
    df_feat['time_of_day'] = df_feat['minute_id'] / 240.0
    
    return df_feat

print("⚙️ Engineering Even-Numbered Features...")
# Combine to engineer features, then split back to avoid code duplication
len_train = len(train_df)
combined_df = pd.concat([train_df, test_df], ignore_index=True)
combined_df = generate_features(combined_df)

train_df = combined_df.iloc[:len_train].reset_index(drop=True)
test_df = combined_df.iloc[len_train:].reset_index(drop=True)

# ==========================================
# 3. STRICT TIME-SERIES SPLIT
# ==========================================
# We train on the first 85% of days, validate on the last 15%
dates = train_df['date_id'].unique()
split_point = int(len(dates) * 0.85)
train_date_limit = dates[split_point]

train_set = train_df[train_df['date_id'] <= train_date_limit]
val_set = train_df[train_df['date_id'] > train_date_limit]

targets = ['target_short', 'target_medium', 'target_long']
drop_cols = targets + ['id', 'date_id', 'minute_id', 'stock_id']
features = [col for col in train_df.columns if col not in drop_cols]

print(f"📊 Using {len(features)} Features.")
print(f"🗓️ Training on Days 0 to {train_date_limit}, Validating on Days {train_date_limit+1} to {dates[-1]}")

# ==========================================
# 4. TRAIN 3 SEPARATE MODELS (CPU Optimized)
# ==========================================
lgb_params = {
    'objective': 'mae',
    'metric': 'mae',
    'device': 'cpu',          # Runs safely on Mac
    'boosting_type': 'gbdt',
    'learning_rate': 0.03,
    'num_leaves': 45,         # Prevents overfitting
    'feature_fraction': 0.6,  # Uses 60% of features per tree (adds randomness)
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'verbose': -1,
    'random_state': 42,
    'n_jobs': -1              # Uses all Mac CPU cores
}

models = {}
val_preds = np.zeros((len(val_set), 3))

for i, target in enumerate(targets):
    print(f"\n🧠 Training Model for {target}...")
    
    trn_data = lgb.Dataset(train_set[features], label=train_set[target])
    val_data = lgb.Dataset(val_set[features], label=val_set[target], reference=trn_data)
    
    model = lgb.train(
        lgb_params,
        trn_data,
        valid_sets=[val_data],
        num_boost_round=1500,
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    models[target] = model
    val_preds[:, i] = model.predict(val_set[features])
    
    mae = mean_absolute_error(val_set[target], val_preds[:, i])
    print(f"✅ Validation MAE for {target}: {mae:.6f}")

# Weighted Validation Score
weights = [0.1, 0.3, 0.6]
wmae = np.sum(np.mean(np.abs(val_set[targets].values - val_preds), axis=0) * weights)
print(f"\n🏆 Overall Weighted Validation MAE: {wmae:.6f}")

# ==========================================
# 5. INFERENCE & THE "MAGIC" POST-PROCESSING
# ==========================================
print("\n🔮 Predicting on Test Set...")
test_preds = np.zeros((len(test_df), 3))

for i, target in enumerate(targets):
    test_preds[:, i] = models[target].predict(test_df[features])

# --- THE MAGIC TRICKS TO BEAT 0.8031 ---

# Trick A: Zero-Mean Centering (Fixes the 0.9036 directional bias error)
test_preds[:, 0] -= test_preds[:, 0].mean()
test_preds[:, 1] -= test_preds[:, 1].mean()
test_preds[:, 2] -= test_preds[:, 2].mean()

# Trick B: Safety Shrinkage (Pulls predictions toward the zero-baseline to minimize risk)
# We trust our model, but we blend it 60% Model / 40% Zero to reduce variance.
test_preds *= 0.6 

# ==========================================
# 6. SAVE SUBMISSION
# ==========================================
submission = pd.DataFrame({
    'id': test_df['id'],
    'target_short': test_preds[:, 0],
    'target_medium': test_preds[:, 1],
    'target_long': test_preds[:, 2]
})

submission.to_csv('submission_lgbm_final.csv', index=False)
print("\n🎉 submission_lgbm_final.csv successfully created!")
print(submission.describe())