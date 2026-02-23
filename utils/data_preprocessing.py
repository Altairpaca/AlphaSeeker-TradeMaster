
# utils/data_preprocessing.py
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List

MAX_MINUTES_PER_DAY = 390  # 可根据数据调整


def expanding_normalize_with_frozen_stats(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    date_col: str = "date_id",
    time_col: str = "minute_id",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """训练集 expanding 标准化，测试集用冻结统计量"""
    train_df = train_df.copy()
    test_df = test_df.copy()

    # 1. 处理 inf/-inf
    for df in [train_df, test_df]:
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

    # 2. 冷启动过滤：基于所有特征的 first valid date
    first_valid_dates = []
    for col in feature_cols:
        valid_rows = train_df[train_df[col].notna()]
        if len(valid_rows) > 0:
            first_valid_dates.append(valid_rows[date_col].min())
    
    if first_valid_dates:
        cutoff_date = min(first_valid_dates)
        train_df = train_df[train_df[date_col] >= cutoff_date].reset_index(drop=True)
        print(f"✅ 冷启动过滤: 保留 date_id >= {cutoff_date}")
    else:
        raise ValueError("训练集中所有特征全为 NaN！")

    # 3. 按时间排序
    train_df = train_df.sort_values([date_col, time_col]).reset_index(drop=True)

    # 4. 填补训练集缺失（前向填充 + 0兜底）
    train_df[feature_cols] = train_df[feature_cols].fillna(method="ffill").fillna(0.0)

    # 5. Expanding 标准化
    train_stats = {}
    for col in feature_cols:
        x = train_df[col]
        exp_mean = x.expanding().mean()
        exp_std = x.expanding().std().clip(lower=1e-8)
        exp_std.iloc[0] = 1.0
        train_df[col] = (x - exp_mean) / exp_std
        train_stats[col] = {"mean": exp_mean.iloc[-1], "std": exp_std.iloc[-1]}

    # 6. 测试集：填补 + 标准化（用冻结 stats）
    for col in feature_cols:
        fill_val = train_stats[col]["mean"]
        test_df[col] = test_df[col].fillna(fill_val)
        mean_val = train_stats[col]["mean"]
        std_val = train_stats[col]["std"]
        test_df[col] = (test_df[col] - mean_val) / std_val

    return train_df, test_df, train_stats


def add_minute_cyclic_features(df: pd.DataFrame) -> pd.DataFrame:
    """添加 minute_sin / minute_cos，不标准化"""
    df = df.copy()
    df["minute_sin"] = np.sin(2 * np.pi * df["minute_id"] / MAX_MINUTES_PER_DAY)
    df["minute_cos"] = np.cos(2 * np.pi * df["minute_id"] / MAX_MINUTES_PER_DAY)
    return df