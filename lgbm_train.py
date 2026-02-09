# lgbm_train_v5.py
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ==============================
# 配置（适配 V5）
# ==============================
DATA_DIR = 'processed_v5'  # ← 改为 V5 输出目录
TRAIN_PATH = os.path.join(DATA_DIR, 'train_processed_v5.csv')
TEST_PATH = os.path.join(DATA_DIR, 'test_processed_v5.csv')
SUBMISSION_MODEL_PATH = 'submission_model.csv'
SUBMISSION_LONG_ZERO_PATH = 'submission_long_zero.csv'

TARGETS = ['target_short', 'target_medium', 'target_long']
RANDOM_STATE = 42

# 标签放大因子（关键！）
SCALE_FACTOR = {
    'target_short': 1000,
    'target_medium': 100,
    'target_long': 10
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
    'gpu_device_id': 0,  # Kaggle T4/P100 通常 device_id=0
}

PARAMS_PER_TARGET = {
    'target_short': {'num_leaves': 128, 'learning_rate': 0.01, 'min_data_in_leaf': 20},
    'target_medium': {'num_leaves': 128, 'learning_rate': 0.015, 'min_data_in_leaf': 50},
    'target_long': {'num_leaves': 256, 'learning_rate': 0.02, 'min_data_in_leaf': 100},
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
# 训练函数
# ==============================
def train_model(X_train, y_train, X_val, y_val, target_name, base_params, scale_factor):
    print(f"\n[INFO] Training model for {target_name} (scale={scale_factor})...")

    y_train_scaled = y_train * scale_factor
    y_val_scaled = y_val * scale_factor

    train_data = lgb.Dataset(X_train, label=y_train_scaled)
    val_data = lgb.Dataset(X_val, label=y_val_scaled, reference=train_data)

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

    pred_val_scaled = model.predict(X_val)
    pred_val = pred_val_scaled / scale_factor
    mae = mean_absolute_error(y_val, pred_val)
    print(f"✅ Validation MAE ({target_name}): {mae:.6f}")
    return model, mae


# ==============================
# 主流程：包含 long=0 对比实验
# ==============================
def main():
    # 1. 加载并划分数据
    df_train, df_val, df_holdout = load_and_split_data(TRAIN_PATH)

    # 2. 准备训练/验证数据
    X_train, y_train = prepare_features(df_train, is_train=True)
    X_val, y_val = prepare_features(df_val, is_train=True)

    # 3. 训练三个模型
    models = {}
    for i, target in enumerate(TARGETS):
        y_tr = y_train[:, i]
        y_v = y_val[:, i]
        model, _ = train_model(X_train, y_tr, X_val, y_v, target, BASE_PARAMS, SCALE_FACTOR[target])
        models[target] = model

    # 4. Hold-out 评估
    X_hold, y_hold = prepare_features(df_holdout, is_train=True)
    print("\n[Hold-out Test Evaluation]")

    mae_dict = {}
    predictions_hold = {}
    for i, target in enumerate(TARGETS):
        pred_scaled = models[target].predict(X_hold)
        pred = pred_scaled / SCALE_FACTOR[target]
        predictions_hold[target] = pred
        mae = mean_absolute_error(y_hold[:, i], pred)
        mae_dict[target] = mae
        print(f"  Hold-out MAE ({target}): {mae:.6f}")

    # 计算原始 WMAE（三个都用模型）
    weights = {'target_short': 0.5, 'target_medium': 0.3, 'target_long': 0.2}
    wmae_model = sum(weights[t] * mae_dict[t] for t in TARGETS)
    print(f"\n🎯 WMAE (all modeled): {wmae_model:.6f}")

    # === 关键对比：将 target_long 设为 0 ===
    pred_long_zero = np.zeros_like(predictions_hold['target_long'])
    mae_long_zero = mean_absolute_error(y_hold[:, 2], pred_long_zero)  # 第2列是 target_long
    wmae_long_zero = (
        weights['target_short'] * mae_dict['target_short'] +
        weights['target_medium'] * mae_dict['target_medium'] +
        weights['target_long'] * mae_long_zero
    )
    print(f"🎯 WMAE (long=0):        {wmae_long_zero:.6f}")
    print(f"📈 Improvement:          {wmae_model - wmae_long_zero:.6f} (positive = better)")

    # 决策：是否使用 long=0？
    use_long_zero = wmae_long_zero < wmae_model
    print(f"\n💡 Recommendation: {'Use target_long=0' if use_long_zero else 'Keep model prediction'}")

    # 5. 预测比赛测试集
    df_test = pd.read_csv(TEST_PATH)
    X_test, test_ids = prepare_features(df_test, is_train=False)

    # 模型预测
    pred_short = models['target_short'].predict(X_test) / SCALE_FACTOR['target_short']
    pred_medium = models['target_medium'].predict(X_test) / SCALE_FACTOR['target_medium']
    pred_long_model = models['target_long'].predict(X_test) / SCALE_FACTOR['target_long']

    # 6. 保存两个提交文件
    # (a) 全模型
    submission_model = pd.DataFrame({
        'id': test_ids,
        'target_short': pred_short,
        'target_medium': pred_medium,
        'target_long': pred_long_model
    })
    submission_model.to_csv(SUBMISSION_MODEL_PATH, index=False)
    print(f"\n✅ Full-model submission saved: {SUBMISSION_MODEL_PATH}")

    # (b) long=0
    submission_long_zero = pd.DataFrame({
        'id': test_ids,
        'target_short': pred_short,
        'target_medium': pred_medium,
        'target_long': 0.0  # ← 关键修改
    })
    submission_long_zero.to_csv(SUBMISSION_LONG_ZERO_PATH, index=False)
    print(f"✅ Long-zero submission saved: {SUBMISSION_LONG_ZERO_PATH}")

    print("\n📤 提交建议：")
    print(f"  • 如果 hold-out 显示 long=0 更好 → 提交 {SUBMISSION_LONG_ZERO_PATH}")
    print(f"  • 否则 → 提交 {SUBMISSION_MODEL_PATH}")
    print("\n📌 注意：Private LB 才是最终标准，但 hold-out 是可靠代理！")


if __name__ == "__main__":
    main()