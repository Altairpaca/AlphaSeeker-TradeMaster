# config.py
import os

# =============
# 路径配置
# =============
DATA_DIR = "dataset"
TRAIN_PATH = os.path.join(DATA_DIR, "train_v2.csv")
TEST_PATH = os.path.join(DATA_DIR, "test_v2.csv")
SUBMISSION_PATH = "submission.csv"

# =============
# 特征与目标
# =============
FEATURE_COLS = [f"feature_{i}" for i in range(1, 31)]
TARGET_COLS = ["target_short", "target_medium", "target_long"]
ID_COLS = ["stock_id", "date_id", "minute_id"]

# =============
# 模型超参数
# =============
MODEL_CONFIG = {
    "hidden_dim": 256,
    "dropout": 0.2,
    "lr": 1e-3,
    "weights": [0.5, 0.3, 0.2],
}

# =============
# 训练配置
# =============
TRAIN_CONFIG = {
    "batch_size": 512,
    "max_epochs": 50,
    "val_split_ratio": 0.9,
    "early_stop_patience": 5,
    "num_workers": 4,
}