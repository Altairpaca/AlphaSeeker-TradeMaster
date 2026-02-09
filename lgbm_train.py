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

# LightGBM 参数（GPU 启用）
BASE_PARAMS = {
    'objective': 'regression_l1',
    'metric': 'mae',
    'device': 'cuda',  # GPU
    'boosting_type': 'gbdt',
    'num_leaves': 256,
    'learning_rate': 0.02,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'min_data_in_leaf': 100,
    'lambda_l1': 1e-2,
    'lambda_l2': 1e-2,
    'verbose': 1,
    'random_state': RANDOM_STATE,
    'gpu_platform_id': 0,
    'gpu_device_id': 2,
}


# ==============================
# 数据加载与划分（时间顺序！）
# ==============================
def load_and_split_data(train_path, val_ratio=0.1, test_ratio=0.1):
    """
    按时间顺序划分：train → val → holdout_test
    假设数据已按时间排序（非常重要！）
    """
    df = pd.read_csv(train_path)
    
    # 确保按时间排序（如有 time_id 或 row_id）
    # 如果没有显式时间列，假设原始顺序就是时间顺序
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
    """提取特征和标签"""
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
def train_model(X_train, y_train, X_val, y_val, target_name, params):
    """训练单个目标的模型"""
    print(f"\n[INFO] Training model for {target_name}...")

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=2000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(100)
        ]
    )

    # 验证集 MAE
    pred_val = model.predict(X_val)
    mae = mean_absolute_error(y_val, pred_val)
    print(f"✅ Validation MAE ({target_name}): {mae:.6f}")
    return model, mae

# ==============================
# 主流程
# ==============================


def main():
    # 1. 加载并划分训练数据（时间顺序）
    df_train, df_val, df_holdout = load_and_split_data(
        TRAIN_PATH, val_ratio=0.15, test_ratio=0.05)

    # 2. 准备训练/验证数据
    X_train, y_train = prepare_features(df_train, is_train=True)
    X_val, y_val = prepare_features(df_val, is_train=True)

    # 3. 训练三个模型
    models = {}
    for i, target in enumerate(TARGETS):
        y_tr = y_train[:, i]
        y_v = y_val[:, i]
        model, _ = train_model(X_train, y_tr, X_val, y_v, target, BASE_PARAMS)
        models[target] = model

    # 4. 在 hold-out 上评估泛化性 + 计算 WMAE
    X_hold, y_hold = prepare_features(df_holdout, is_train=True)
    print("\n[Hold-out Test Evaluation]")

    mae_dict = {}
    for i, target in enumerate(TARGETS):
        pred = models[target].predict(X_hold)
        mae = mean_absolute_error(y_hold[:, i], pred)
        mae_dict[target] = mae
        print(f"  Hold-out MAE ({target}): {mae:.6f}")

    # 计算加权 MAE (WMAE)
    weights = {
        'target_short': 0.5,
        'target_medium': 0.3,
        'target_long': 0.2
    }
    wmae = (
        weights['target_short'] * mae_dict['target_short'] +
        weights['target_medium'] * mae_dict['target_medium'] +
        weights['target_long'] * mae_dict['target_long']
    )
    print(f"\n🎯 Weighted MAE (WMAE) on Hold-out: {wmae:.6f}")

    # 5. 加载比赛测试集并预测
    df_test = pd.read_csv(TEST_PATH)
    X_test, test_ids = prepare_features(df_test, is_train=False)

    predictions = {}
    for target in TARGETS:
        pred = models[target].predict(X_test)
        predictions[target] = pred

    # 6. 保存提交文件
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