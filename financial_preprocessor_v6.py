# financial_preprocessor_v6_4.py
import os
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr

class FinancialPreprocessorV6_4:
    """
    V6.4: 放弃价格代理，直接构建收益率预测特征
    ✅ 对所有原始特征计算多尺度差分
    ✅ 用 target_short 相关性筛选最强信号
    ✅ 彻底清理 NaN
    """
    
    def __init__(self, 
                 data_dir='dataset',
                 output_dir='processed_v6_4',
                 random_state=42):
        self.data_dir = data_dir
        self.output_dir = output_dir
        Path(output_dir).mkdir(exist_ok=True)
        self.random_state = random_state
        
        self.date_col = 'date_id'
        self.minute_col = 'minute_id'
        
        # 所有原始特征（30个）
        self.raw_features = [f'feature_{i}' for i in range(1, 31)]
        
    def _smart_fill_feature(self, series):
        """智能填充（无未来泄露）"""
        s = series.copy()
        first_valid = s.first_valid_index()
        if first_valid is not None and first_valid > s.index[0]:
            s.loc[:first_valid] = s.loc[first_valid]
        s = s.ffill().bfill()
        if s.isna().any():
            s = s.fillna(s.mean())
        return s
    
    def _basic_cleaning(self, df):
        """清洗所有原始特征"""
        df = df.sort_values([self.date_col, self.minute_col]).reset_index(drop=True)
        
        for feat in self.raw_features:
            if feat in df.columns and df[feat].isna().any():
                df[feat] = self._smart_fill_feature(df[feat])
        
        return df
    
    def _create_engineered_features(self, df):
        """构建收益率预测特征（不依赖价格代理）"""
        df = df.copy()
        
        print(f"\n✨ 构建收益率预测特征（对所有原始特征计算差分）...")
        
        # 对每个原始特征计算多尺度差分
        diff_windows = [1, 2, 5, 10]
        for feat in self.raw_features:
            if feat not in df.columns:
                continue
            series = df[feat]
            for window in diff_windows:
                col_name = f'{feat}_diff_{window}'
                df[col_name] = series.diff(window).fillna(0)
                
                # 添加波动率
                vol_col = f'{feat}_vol_{window*2}'
                df[vol_col] = df[col_name].rolling(window*2, min_periods=max(1, window)).std().fillna(0)
        
        # 量价交互（假设 feature_24 是 volume）
        if 'feature_24' in df.columns:
            volume = df['feature_24']
            for feat in ['feature_3', 'feature_4', 'feature_5', 'feature_7', 'feature_13']:
                if feat in df.columns:
                    # 体积加权变化
                    df[f'{feat}_vol_weighted'] = volume * df[f'{feat}_diff_1']
                    # 相关性
                    df[f'{feat}_vol_corr_10'] = (
                        volume.rolling(10, min_periods=5).corr(df[f'{feat}_diff_1'])
                        .fillna(0)
                    )
        
        # 时间特征
        max_min = df[self.minute_col].max()
        df['minute_sin'] = np.sin(2 * np.pi * df[self.minute_col] / max_min)
        df['minute_cos'] = np.cos(2 * np.pi * df[self.minute_col] / max_min)
        
        # 彻底清理
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], 0)
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        print(f"   ✅ 收益率预测特征构建完成")
        return df
    
    def _feature_selection_and_scaling(self, train_df, test_df):
        """严格筛选与 target_short 高相关的特征"""
        train_df['is_train'] = 1
        test_df['is_train'] = 0
        combined = pd.concat([train_df, test_df], ignore_index=True)
        
        exclude = ['id', 'target_short', 'target_medium', 'target_long', 
                  'is_train', self.date_col, self.minute_col, 'stock_id']
        all_features = [c for c in combined.columns if c not in exclude]
        
        # 计算每个特征与 target_short 的相关性
        train_only = combined[combined['is_train'] == 1]
        feature_corr = []
        
        for col in all_features:
            try:
                corr = abs(pearsonr(train_only[col].fillna(0), train_only['target_short'])[0])
                feature_corr.append((col, corr))
            except:
                feature_corr.append((col, 0))
        
        # 选择 top 50 最相关特征 + 原始特征
        feature_corr.sort(key=lambda x: x[1], reverse=True)
        top_features = [col for col, corr in feature_corr[:50]]
        
        # 确保包含原始特征（防止过拟合 engineered 特征）
        final_features = list(set(top_features + self.raw_features))
        final_features = [f for f in final_features if f in combined.columns]
        
        print(f"🔍 保留 {len(final_features)} 个高相关特征（Top 50 + 原始特征）")
        print(f"   🔥 Top 5 相关特征:")
        for i, (col, corr) in enumerate(feature_corr[:5]):
            print(f"      {i+1}. {col}: {corr:.4f}")
        
        # 标准化
        for col in final_features:
            mean_val = train_only[col].mean()
            std_val = train_only[col].std()
            if std_val < 1e-6:
                std_val = 1.0
            combined[col] = (combined[col] - mean_val) / std_val
        
        # 添加 id
        final_columns = ['id'] + final_features + ['target_short', 'target_medium', 'target_long']
        final_columns = [c for c in final_columns if c in combined.columns]
        
        train_final = combined[combined['is_train'] == 1][final_columns]
        test_final = combined[combined['is_train'] == 0][final_columns]
        
        return train_final, test_final
    
    def run(self):
        """主流程"""
        print("="*70)
        print("🚀 金融数据预处理 V6.4 | 收益率预测专用版")
        print("="*70)
        
        train_df = pd.read_csv(os.path.join(self.data_dir, 'train_v2.csv'))
        test_df = pd.read_csv(os.path.join(self.data_dir, 'test_v2.csv'))
        print(f"📊 训练集: {len(train_df):,} 条 | 测试集: {len(test_df):,} 条")
        
        print("\n[1/3] 智能缺失值修复...")
        train_clean = self._basic_cleaning(train_df.copy())
        test_clean = self._basic_cleaning(test_df.copy())
        
        print("\n[2/3] 构建收益率预测特征...")
        train_engineered = self._create_engineered_features(train_clean)
        test_engineered = self._create_engineered_features(test_clean)
        
        print("\n[3/3] 特征筛选与标准化...")
        train_final, test_final = self._feature_selection_and_scaling(
            train_engineered, test_engineered
        )
        
        self._save_processed_data(train_final, test_final)
        self._validate_output(train_final)
        
        return train_final, test_final
    
    def _save_processed_data(self, train_df, test_df):
        train_path = os.path.join(self.output_dir, 'train_processed_v6_4.csv')
        test_path = os.path.join(self.output_dir, 'test_processed_v6_4.csv')
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        print(f"\n✅ 处理完成！")
        print(f"   📁 训练集: {train_path} | 形状: {train_df.shape}")
        print(f"   📁 测试集: {test_path} | 形状: {test_df.shape}")
    
    def _validate_output(self, train_df):
        """关键验证"""
        print("\n🔍 关键验证:")
        missing = train_df.isna().sum().sum()
        print(f"   • 训练集缺失值总数: {missing} {'✅' if missing == 0 else '❌'}")
        
        # 找出与 target_short 最相关的特征
        if 'target_short' in train_df.columns:
            correlations = []
            for col in train_df.columns:
                if col in ['id', 'target_short', 'target_medium', 'target_long']:
                    continue
                try:
                    corr = abs(pearsonr(train_df[col].fillna(0), train_df['target_short'])[0])
                    correlations.append((col, corr))
                except:
                    continue
            
            correlations.sort(key=lambda x: x[1], reverse=True)
            top_corr = correlations[0][1] if correlations else 0
            top_feat = correlations[0][0] if correlations else "N/A"
            
            print(f"   • 最强特征 '{top_feat}' 与 target_short 相关性: {top_corr:.4f} {'🔥' if top_corr > 0.15 else '⚠️'}")


if __name__ == "__main__":
    preprocessor = FinancialPreprocessorV6_4(
        data_dir='dataset',
        output_dir='processed_v6_4',
        random_state=42
    )
    train_df, test_df = preprocessor.run()