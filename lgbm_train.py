# lgbm_train.py
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ==============================
# 配置
# ==============================
DATA_DIR = 'dataset'
TRAIN_PATH = os.path.join(DATA_DIR, 'train_processed.csv')
TEST_PATH = os.path.join(DATA_DIR, 'test_processed.csv')
SUBMISSION_PATH = 'submission.csv'

TARGETS = ['target_short', 'target_medium', 'target_long']
RANDOM_STATE = 42

# 标签放大因子（关键！）
SCALE_FACTOR = {
    'target_short': 1000,   # 放大 1000 倍 → ~2.8
    'target_medium': 100,   # 放大 100 倍
    'target_long': 10       # 放大 10 倍
}

# 基础参数（GPU）
BASE_PARAMS = {
    'objective': 'regression_l1',
    'metric': 'mae',
    'device': 'cuda',
    'boosting_type': 'gbdt',
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'lambda_l1': 1e-2,
    'lambda_l2': 1e-2,
    'verbose': -1,
    'random_state': RANDOM_STATE,
    'gpu_platform_id': 0,
    'gpu_device_id': 2,  # 请根据 nvidia-smi 调整
}

# 每个目标的定制参数
PARAMS_PER_TARGET = {
    'target_short': {
        'num_leaves': 128,
        'learning_rate': 0.01,
        'min_data_in_leaf': 20,      # 更小，捕捉微弱信号
    },
    'target_medium': {
        'num_leaves': 128,
        'learning_rate': 0.015,
        'min_data_in_leaf': 50,
    },
    'target_long': {
        'num_leaves': 256,
        'learning_rate': 0.02,
        'min_data_in_leaf': 100,
    }
}


# ==============================
# 数据加载与划分（时间顺序！）
# ==============================
def load_and_split_data(train_path, val_ratio=0.15, test_ratio=0.05):
    df = pd.read_csv(train_path)
    n = len(df)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)
    n_train = n - n_val - n_test

    if n_train <= 0:
        raise ValueError("Dataset too small for the given ratios.")

    df_train = df.iloc[:n_train].copy()
    df_val = df.iloc[n_train:n_train + n_val].copy()
    df_holdout = df.iloc[n_train + n_val:].copy()

    print(f"Temporal split:")
    print(f"  Train:     {len(df_train)}")
    print(f"  Val:       {len(df_val)}")
    print(f"  Hold-out:  {len(df_holdout)}")

    return df_train, df_val, df_holdout


def prepare_features(df, is_train=True):
    if is_train:
        y = df[TARGETS].values
        X = df.drop(columns=TARGETS + ['id'], errors='ignore').values
        return X, y
    else:
        ids = df['id'].values
        X = df.drop(columns=['id'], errors='ignore').values
        return X, ids


# ==============================
# 训练函数（带 Scale Trick）
# ==============================
def train_model(X_train, y_train, X_val, y_val, target_name, base_params, scale_factor):
    print(
        f"\n[INFO] Training model for {target_name} (scale={scale_factor})...")

    # 放大标签
    y_train_scaled = y_train * scale_factor
    y_val_scaled = y_val * scale_factor

    train_data = lgb.Dataset(X_train, label=y_train_scaled)
    val_data = lgb.Dataset(X_val, label=y_val_scaled, reference=train_data)

    # 合并参数
    params = base_params.copy()
    params.update(PARAMS_PER_TARGET[target_name])

    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=2000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=True),
            lgb.log_evaluation(50)
        ]
    )

    # 验证集预测（缩回原始尺度）
    pred_val_scaled = model.predict(X_val)
    pred_val = pred_val_scaled / scale_factor
    mae = mean_absolute_error(y_val, pred_val)
    print(f"✅ Validation MAE ({target_name}): {mae:.6f}")
    return model, mae


# ==============================
# 主流程
# ==============================
def main():
    # 1. 加载并划分数据
    df_train, df_val, df_holdout = load_and_split_data(TRAIN_PATH)

    # 2. 准备数据
    X_train, y_train = prepare_features(df_train, is_train=True)
    X_val, y_val = prepare_features(df_val, is_train=True)

    # 3. 训练三个模型
    models = {}
    for i, target in enumerate(TARGETS):
        y_tr = y_train[:, i]
        y_v = y_val[:, i]
        model, _ = train_model(
            X_train, y_tr, X_val, y_v, target,
            BASE_PARAMS, SCALE_FACTOR[target]
        )
        models[target] = model

    # 4. Hold-out 评估 + WMAE
    X_hold, y_hold = prepare_features(df_holdout, is_train=True)
    print("\n[Hold-out Test Evaluation]")

    mae_dict = {}
    for i, target in enumerate(TARGETS):
        pred_scaled = models[target].predict(X_hold)
        pred = pred_scaled / SCALE_FACTOR[target]
        mae = mean_absolute_error(y_hold[:, i], pred)
        mae_dict[target] = mae
        print(f"  Hold-out MAE ({target}): {mae:.6f}")

    # 计算 WMAE
    weights = {'target_short': 0.5, 'target_medium': 0.3, 'target_long': 0.2}
    wmae = sum(weights[t] * mae_dict[t] for t in TARGETS)
    print(f"\n🎯 Weighted MAE (WMAE) on Hold-out: {wmae:.6f}")

    # 5. 预测比赛测试集
    df_test = pd.read_csv(TEST_PATH)
    X_test, test_ids = prepare_features(df_test, is_train=False)

    predictions = {}
    for target in TARGETS:
        pred_scaled = models[target].predict(X_test)
        pred = pred_scaled / SCALE_FACTOR[target]
        predictions[target] = pred

    # 6. 保存提交
    submission_df = pd.DataFrame({
        'id': test_ids,
        'target_short': predictions['target_short'],
        'target_medium': predictions['target_medium'],
        'target_long': predictions['target_long']
    })
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"\n✅ Submission saved to {SUBMISSION_PATH}")
    print("Submission shape:", submission_df.shape)
    print(submission_df.head())


if __name__ == "__main__":
    main()
