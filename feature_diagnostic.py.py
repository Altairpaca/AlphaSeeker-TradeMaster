# feature_diagnostic.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 配置
PROCESSED_TRAIN_PATH = 'processed_v4/train_processed_v4.csv'
OUTPUT_DIR = 'analysis'
TARGETS = ['target_short', 'target_medium', 'target_long']
FEATURE_PREFIXES = ['feature_', 'f11_f18', 'f21_f25']  # 包含原始特征和交互项

Path(OUTPUT_DIR).mkdir(exist_ok=True)

def load_data():
    df = pd.read_csv(PROCESSED_TRAIN_PATH)
    # 确保按时间排序
    if 'date_id' in df.columns and 'minute_id' in df.columns:
        df = df.sort_values(['date_id', 'minute_id']).reset_index(drop=True)
    return df

def analyze_missing_and_stats(df):
    """分析缺失值和基本统计"""
    feature_cols = [col for col in df.columns 
                   if any(col.startswith(p) for p in FEATURE_PREFIXES)]
    
    stats = []
    for col in feature_cols:
        missing_pct = df[col].isnull().mean() * 100
        if missing_pct > 0:
            print(f"⚠️  {col} has {missing_pct:.2f}% missing values!")
        
        stats.append({
            'feature': col,
            'missing_pct': missing_pct,
            'mean': df[col].mean(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'skew': df[col].skew(),
            'kurtosis': df[col].kurtosis()
        })
    
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(f'{OUTPUT_DIR}/feature_statistics.csv', index=False)
    print(f"\n📊 Feature statistics saved to {OUTPUT_DIR}/feature_statistics.csv")
    return stats_df

def compute_lag_correlations(df, max_lag=5):
    """
    计算每个特征与三个目标的滞后相关性（lag=1 到 max_lag）
    这是判断特征是否有预测力的核心指标！
    """
    feature_cols = [col for col in df.columns 
                   if any(col.startswith(p) for p in FEATURE_PREFIXES)]
    
    results = []
    
    for target in TARGETS:
        print(f"\n🔍 Computing lag correlations for {target}...")
        for feat in feature_cols:
            for lag in range(1, max_lag + 1):
                # 特征滞后 lag 步，与当前目标对齐
                corr = df[feat].shift(lag).corr(df[target])
                results.append({
                    'feature': feat,
                    'target': target,
                    'lag': lag,
                    'correlation': corr if pd.notna(corr) else 0
                })
    
    corr_df = pd.DataFrame(results)
    corr_df['abs_corr'] = corr_df['correlation'].abs()
    
    # 保存完整结果
    corr_df.to_csv(f'{OUTPUT_DIR}/lag_correlations.csv', index=False)
    
    # 找出每个目标 top 20 最相关特征（按 |corr|）
    top_features = {}
    for target in TARGETS:
        top = corr_df[corr_df['target'] == target].nlargest(20, 'abs_corr')
        top_features[target] = top[['feature', 'lag', 'correlation']].values.tolist()
        print(f"\n🏆 Top predictive features for {target}:")
        for i, (feat, lag, corr) in enumerate(top_features[target][:10]):
            print(f"  {i+1:2d}. {feat} (lag={lag}) → corr={corr:.4f}")
    
    return corr_df, top_features

def evaluate_rolling_features(df):
    """检查 rolling 特征是否比原始特征更相关"""
    rolling_cols = [col for col in df.columns if 'rolling_mean' in col]
    original_map = {
        'feature_11_rolling_mean_10': 'feature_11',
        'feature_18_rolling_mean_10': 'feature_18',
        'feature_21_rolling_mean_10': 'feature_21',
        'feature_25_rolling_mean_10': 'feature_25',
    }
    
    results = []
    for roll_col, orig_col in original_map.items():
        if roll_col in df.columns and orig_col in df.columns:
            for target in TARGETS:
                corr_orig = df[orig_col].shift(1).corr(df[target])
                corr_roll = df[roll_col].shift(1).corr(df[target])
                improvement = corr_roll - corr_orig
                results.append({
                    'feature': orig_col,
                    'rolling_feature': roll_col,
                    'target': target,
                    'orig_corr': corr_orig,
                    'rolling_corr': corr_roll,
                    'improvement': improvement
                })
    
    roll_df = pd.DataFrame(results)
    roll_df.to_csv(f'{OUTPUT_DIR}/rolling_feature_evaluation.csv', index=False)
    print(f"\n📈 Rolling feature evaluation saved to {OUTPUT_DIR}/rolling_feature_evaluation.csv")
    
    # 打印是否有提升
    for _, row in roll_df.iterrows():
        if row['improvement'] > 0.01:
            print(f"✅ {row['rolling_feature']} improves correlation with {row['target']} by {row['improvement']:.4f}")
    
    return roll_df

def suggest_feature_selection(corr_df, threshold=0.01):
    """建议保留哪些特征：至少在一个目标上 |lag-1 corr| > threshold"""
    lag1_corr = corr_df[corr_df['lag'] == 1]
    useful_features = set()
    
    for target in TARGETS:
        target_corr = lag1_corr[lag1_corr['target'] == target]
        useful = target_corr[target_corr['abs_corr'] > threshold]['feature'].unique()
        useful_features.update(useful)
        print(f"\n💡 Features with |lag-1 corr| > {threshold} for {target}: {len(useful)}")
    
    all_features = set(corr_df['feature'].unique())
    useless_features = all_features - useful_features
    
    print(f"\n🗑️  Suggested to DROP ({len(useless_features)} features):")
    for feat in sorted(useless_features)[:20]:
        print(f"  - {feat}")
    if len(useless_features) > 20:
        print(f"  ... and {len(useless_features)-20} more")
    
    # 保存建议
    pd.Series(sorted(useful_features)).to_csv(
        f'{OUTPUT_DIR}/suggested_keep_features.txt', index=False, header=False)
    pd.Series(sorted(useless_features)).to_csv(
        f'{OUTPUT_DIR}/suggested_drop_features.txt', index=False, header=False)
    
    return useful_features, useless_features

def main():
    print("🔍 Starting feature diagnostic analysis...")
    df = load_data()
    
    # 1. 基础统计
    stats_df = analyze_missing_and_stats(df)
    
    # 2. 滞后相关性（最关键！）
    corr_df, top_features = compute_lag_correlations(df, max_lag=3)
    
    # 3. Rolling 特征评估
    roll_df = evaluate_rolling_features(df)
    
    # 4. 特征筛选建议
    useful, useless = suggest_feature_selection(corr_df, threshold=0.005)
    
    print(f"\n🎉 Analysis complete! Check '{OUTPUT_DIR}' for detailed reports.")
    
    # 可选：绘制 top 相关性热图
    try:
        plt.figure(figsize=(12, 8))
        pivot = corr_df[corr_df['lag']==1].pivot(index='feature', columns='target', values='correlation')
        # 只显示 top 30 最相关的特征
        top_feats = corr_df[corr_df['lag']==1].groupby('feature')['abs_corr'].max().nlargest(30).index
        sns.heatmap(pivot.loc[top_feats].fillna(0), center=0, cmap='RdBu_r', annot=True, fmt='.3f')
        plt.title('Lag-1 Correlation: Top 30 Features vs Targets')
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/lag1_correlation_heatmap.png', dpi=150)
        print(f"🖼️  Correlation heatmap saved to {OUTPUT_DIR}/lag1_correlation_heatmap.png")
    except Exception as e:
        print(f"⚠️  Could not generate heatmap: {e}")

if __name__ == "__main__":
    main()