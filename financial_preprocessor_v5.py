# financial_preprocessor_v5.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class FinancialPreprocessorV5:
    """
    V5 核心改进：
    ✅ 移除所有无效收益率/动量特征（相关性=0）
    ✅ 仅保留高信号原始特征 + 有效衍生（滚动均值/波动率/交互）
    ✅ 新增特征重要性可视化（自动评估特征质量）
    ✅ 强化单股票场景适配（完全独立处理训练/测试集）
    """
    
    def __init__(self, cold_start_day=4, save_dir='processed_v5'):
        self.cold_start_day = cold_start_day
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.fill_values = {}
        # 仅保留诊断验证有效的原始特征（来自 suggested_keep_features.txt）
        self.keep_original_features = [
            'feature_3', 'feature_4', 'feature_5', 'feature_6', 'feature_7', 'feature_8',
            'feature_11', 'feature_12', 'feature_13', 'feature_14', 'feature_16',
            'feature_18', 'feature_19', 'feature_20', 'feature_21', 'feature_23',
            'feature_24', 'feature_25', 'feature_26', 'feature_27', 'feature_28',
            'feature_29', 'feature_30'
        ]
        self.price_proxy_features = ['feature_13', 'feature_14', 'feature_24']  # 波动率计算用
        self.rolling_features = ['feature_18', 'feature_21', 'feature_25']      # 滚动均值用
    
    def _load_and_prepare(self, df, is_train=True):
        """基础清理：替换inf，确保时间排序"""
        df = df.copy()
        # 仅处理需要的特征列（减少内存占用）
        all_cols = self.keep_original_features + ['id', 'date_id', 'minute_id', 'stock_id']
        if is_train:
            all_cols += ['target_short', 'target_medium', 'target_long']
        df = df[[c for c in all_cols if c in df.columns]]
        
        # 安全替换inf
        df[self.keep_original_features] = df[self.keep_original_features].replace([np.inf, -np.inf], np.nan)
        # 严格按时间排序
        if 'date_id' in df.columns and 'minute_id' in df.columns:
            df = df.sort_values(['date_id', 'minute_id']).reset_index(drop=True)
        df['_is_train'] = 1 if is_train else 0
        return df
    
    def _handle_cold_start_missing(self, df):
        """仅处理训练集冷启动缺失（前cold_start_day天）"""
        df_processed = df.copy()
        cold_start_mask = (df_processed['_is_train'] == 1) & (df_processed['date_id'] < self.cold_start_day)
        
        if cold_start_mask.any():
            print(f"❄️  处理训练集冷启动缺失: 前 {self.cold_start_day} 天 (共 {cold_start_mask.sum()} 条)")
            for feat in self.keep_original_features:
                if feat == 'feature_11': 
                    continue
                valid_data = df_processed[
                    (df_processed['date_id'] >= self.cold_start_day) & 
                    df_processed[feat].notna()
                ]
                if len(valid_data) > 0:
                    fill_val = valid_data.iloc[0][feat]
                    self.fill_values[feat] = fill_val
                    feat_mask = cold_start_mask & df_processed[feat].isna()
                    if feat_mask.any():
                        df_processed.loc[feat_mask, feat] = fill_val
                        print(f"  ✓ {feat}: 用 date_id={self.cold_start_day} 的值 {fill_val:.4f} 填充 {feat_mask.sum()} 处")
        return df_processed
    
    def _handle_feature_11_missing(self, df):
        """精准修复feature_11：系统性缺失(237-238) + 随机缺失插值"""
        df_processed = df.copy()
        original_missing = df_processed['feature_11'].isna()
        systemic_mask = original_missing & df_processed['minute_id'].isin([237, 238])
        
        # 修复系统性缺失：用同天minute=236的值
        if systemic_mask.any():
            fill_map = df_processed[df_processed['minute_id'] == 236].set_index('date_id')['feature_11']
            for idx in df_processed[systemic_mask].index:
                date_id = df_processed.at[idx, 'date_id']
                if date_id in fill_map.index and pd.notna(fill_map[date_id]):
                    df_processed.at[idx, 'feature_11'] = fill_map[date_id]
            print(f"🔧 修复 feature_11 系统性缺失: {systemic_mask.sum()} 处 (分钟237-238)")
        
        # 剩余缺失：按天线性插值
        if df_processed['feature_11'].isna().any():
            df_processed['feature_11'] = df_processed.groupby('date_id')['feature_11'].transform(
                lambda x: x.interpolate(method='linear', limit_direction='both')
            )
            remaining = df_processed['feature_11'].isna().sum()
            if remaining > 0:
                median_val = df_processed['feature_11'].median()
                df_processed['feature_11'] = df_processed['feature_11'].fillna(median_val)
                print(f"⚠️  剩余 {remaining} 处用中位数 {median_val:.4f} 填充")
        
        # 保留缺失指示
        df_processed['f11_sys_missing'] = systemic_mask.astype(int)
        df_processed['f11_rand_missing'] = (original_missing & ~systemic_mask).astype(int)
        print(f"📌 创建缺失指示: f11_sys_missing={systemic_mask.sum()}, f11_rand_missing={(original_missing & ~systemic_mask).sum()}")
        return df_processed
    
    def _handle_other_missing(self, df):
        """健壮性处理其他特征随机缺失"""
        df_processed = df.copy()
        other_features = [f for f in self.keep_original_features if f != 'feature_11']
        
        for feat in other_features:
            mask = (df_processed['date_id'] >= self.cold_start_day) & df_processed[feat].isna()
            if mask.any():
                df_processed[feat] = df_processed.groupby('date_id')[feat].transform(
                    lambda x: x.fillna(method='ffill').fillna(method='bfill')
                )
                if df_processed[feat].isna().sum() > 0:
                    df_processed[feat] = df_processed[feat].fillna(df_processed[feat].median())
                print(f"🔧 修复 {feat} 随机缺失: {mask.sum()} 处")
        return df_processed
    
    def _create_volatility_features(self, df):
        """仅创建波动率特征（诊断显示微弱有效）"""
        df_engineered = df.copy()
        
        # 单股票场景：直接按时间顺序计算（无需 groupby）
        for feat in self.price_proxy_features:
            if feat not in df_engineered.columns:
                continue
            
            # 计算分钟级收益率（前向差分）
            rtn = df_engineered[feat].pct_change(1)
            
            # 计算过去10分钟收益率的标准差（波动率）
            vol = rtn.rolling(window=10, min_periods=1).std()
            
            # 填充开头的 NaN（用0或后向填充）
            df_engineered[f'{feat}_vol_10'] = vol.fillna(0)
        
        print(f"📉 创建波动率特征: {self.price_proxy_features}")
        return df_engineered
    
    def _create_rolling_features(self, df):
        """创建滚动均值特征（诊断验证有效）"""
        df_engineered = df.copy()
        for feat in self.rolling_features:
            if feat in df_engineered.columns:
                # 单股票：直接滚动
                df_engineered[f'{feat}_roll_mean_10'] = (
                    df_engineered[feat]
                    .rolling(window=10, min_periods=1)
                    .mean()
                    .fillna(0)
                )
        print(f"🌀 创建滚动均值特征: {self.rolling_features}")
        return df_engineered
    
    def _create_interactions(self, df):
        """创建交互特征（诊断验证有效）"""
        df_engineered = df.copy()
        if 'feature_11' in df_engineered.columns and 'feature_18' in df_engineered.columns:
            df_engineered['f11_f18'] = df_engineered['feature_11'] * df_engineered['feature_18']
        if 'feature_21' in df_engineered.columns and 'feature_25' in df_engineered.columns:
            df_engineered['f21_f25'] = df_engineered['feature_21'] * df_engineered['feature_25']
        print("🔀 创建交互特征: f11_f18, f21_f25")
        return df_engineered
    
    def _create_time_features(self, df):
        """轻量级时间特征"""
        df_engineered = df.copy()
        df_engineered['is_near_close'] = (df_engineered['minute_id'] >= 230).astype(int)
        df_engineered['trading_progress'] = df_engineered['minute_id'] / 239.0
        return df_engineered
    
    def _final_sanity_check(self, df):
        """最终检查：填充剩余缺失"""
        df_final = df.copy()
        num_cols = df_final.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if df_final[col].isna().any():
                fill_val = df_final[col].median()
                df_final[col] = df_final[col].fillna(fill_val)
                if df[col].isna().sum() > 0:
                    print(f"⚠️  最终修复 {col} 剩余缺失: {df[col].isna().sum()} 处 -> 用中位数 {fill_val:.4f}")
        return df_final
    
    def _remove_unnecessary_columns(self, df, is_test=False):
        """移除无信息量列"""
        cols_to_drop = ['stock_id', 'date_id', 'minute_id', '_is_train']
        if is_test:
            cols_to_drop += ['target_short', 'target_medium', 'target_long']
        return df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
    
    def _visualize_feature_importance(self, train_df, top_k=20):
        """特征重要性可视化（使用随机森林）"""
        print("\n🔍 正在计算特征重要性...")
        # 准备数据
        feature_cols = [c for c in train_df.columns 
                       if c not in ['id', 'target_short', 'target_medium', 'target_long']]
        X = train_df[feature_cols]
        y_short = train_df['target_short']
        y_medium = train_df['target_medium']
        
        # 训练两个模型（short + medium）
        rf_short = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf_medium = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        
        # 使用部分数据加速（10%）
        sample_size = min(10000, len(X))
        idx = np.random.choice(len(X), size=sample_size, replace=False)
        
        rf_short.fit(X.iloc[idx], y_short.iloc[idx])
        rf_medium.fit(X.iloc[idx], y_medium.iloc[idx])
        
        # 获取重要性
        imp_short = pd.Series(rf_short.feature_importances_, index=feature_cols).sort_values(ascending=False)
        imp_medium = pd.Series(rf_medium.feature_importances_, index=feature_cols).sort_values(ascending=False)
        
        # 绘图
        plt.figure(figsize=(16, 10))
        
        # Short importance
        plt.subplot(2, 1, 1)
        sns.barplot(x=imp_short[:top_k].values, y=imp_short[:top_k].index, palette="viridis")
        plt.title(f'Top {top_k} Features for target_short (Random Forest Importance)', fontsize=14)
        plt.xlabel('Importance')
        
        # Medium importance
        plt.subplot(2, 1, 2)
        sns.barplot(x=imp_medium[:top_k].values, y=imp_medium[:top_k].index, palette="magma")
        plt.title(f'Top {top_k} Features for target_medium (Random Forest Importance)', fontsize=14)
        plt.xlabel('Importance')
        
        plt.tight_layout()
        viz_path = self.save_dir / 'feature_importance.png'
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ 特征重要性图保存至: {viz_path}")
        
        # 保存重要性分数
        imp_df = pd.DataFrame({
            'feature': feature_cols,
            'importance_short': rf_short.feature_importances_,
            'importance_medium': rf_medium.feature_importances_
        }).sort_values('importance_short', ascending=False)
        imp_df.to_csv(self.save_dir / 'feature_importance_scores.csv', index=False)
        print(f"📊 特征重要性分数保存至: {self.save_dir / 'feature_importance_scores.csv'}")
        
        # 打印Top 10
        print("\n🏆 Top 10 特征 (target_short):")
        for i, (feat, imp) in enumerate(imp_short.head(10).items(), 1):
            print(f"  {i}. {feat}: {imp:.4f}")
        
        return imp_df
    
    def fit_transform(self, train_df, test_df=None, save=True, visualize=True):
        """
        完全独立处理训练/测试集（应对时间不连续）
        """
        print("="*70)
        print("🚀 金融数据预处理 V5 | 精简高效版")
        print("="*70)
        
        # =============== 1. 分别加载与准备 ===============
        print("\n[1/2] 处理训练集")
        train_proc = self._load_and_prepare(train_df, is_train=True)
        print(f"   📊 训练集: {len(train_proc)} 条 | 保留原始特征: {len(self.keep_original_features)} 个")
        
        if test_df is not None:
            # 确保测试集有目标列占位
            for tgt in ['target_short', 'target_medium', 'target_long']:
                if tgt not in test_df.columns:
                    test_df[tgt] = np.nan
            test_proc = self._load_and_prepare(test_df, is_train=False)
            print(f"\n[2/2] 处理测试集")
            print(f"   📊 测试集: {len(test_proc)} 条 | 时间不连续: 训练集结束于 {train_proc['date_id'].max()}, 测试集开始于 {test_proc['date_id'].min()}")
        else:
            test_proc = None
        
        # =============== 2. 独立处理训练集 ===============
        print("\n" + "-"*70)
        print("🔧 训练集处理流程")
        print("-"*70)
        train_proc = self._handle_cold_start_missing(train_proc)
        train_proc = self._handle_feature_11_missing(train_proc)
        train_proc = self._handle_other_missing(train_proc)
        train_proc = self._create_volatility_features(train_proc)     # 仅波动率！
        train_proc = self._create_rolling_features(train_proc)        # 滚动均值
        train_proc = self._create_interactions(train_proc)            # 交互
        train_proc = self._create_time_features(train_proc)           # 时间特征
        train_proc = self._final_sanity_check(train_proc)
        train_final = self._remove_unnecessary_columns(train_proc, is_test=False)
        
        # =============== 3. 独立处理测试集 ===============
        if test_proc is not None:
            print("\n" + "-"*70)
            print("🔧 测试集处理流程（完全独立）")
            print("-"*70)
            test_proc = self._handle_cold_start_missing(test_proc)    # 实际无操作
            test_proc = self._handle_feature_11_missing(test_proc)
            test_proc = self._handle_other_missing(test_proc)
            test_proc = self._create_volatility_features(test_proc)
            test_proc = self._create_rolling_features(test_proc)
            test_proc = self._create_interactions(test_proc)
            test_proc = self._create_time_features(test_proc)
            test_proc = self._final_sanity_check(test_proc)
            test_final = self._remove_unnecessary_columns(test_proc, is_test=True)
        else:
            test_final = None
        
        # =============== 4. 保存与可视化 ===============
        if save:
            train_path = self.save_dir / 'train_processed_v5.csv'
            train_final.to_csv(train_path, index=False)
            print(f"\n✅ 训练集保存: {train_path} | 形状: {train_final.shape}")
            
            if test_final is not None:
                test_path = self.save_dir / 'test_processed_v5.csv'
                test_final.to_csv(test_path, index=False)
                print(f"✅ 测试集保存: {test_path} | 形状: {test_final.shape} (已移除目标列)")
            
            # 保存特征列表
            feature_cols = [c for c in train_final.columns if c not in ['id', 'target_short', 'target_medium', 'target_long']]
            pd.Series(sorted(feature_cols)).to_csv(self.save_dir / 'feature_list_v5.txt', index=False, header=False)
            print(f"📋 特征列表保存: {len(feature_cols)} 个特征")
        
        # =============== 5. 特征重要性可视化 ===============
        if visualize and save:
            self._visualize_feature_importance(train_final)
        
        # =============== 6. 关键验证 ===============
        print("\n" + "="*70)
        print("🔍 处理验证")
        print("="*70)
        # 验证ID列已移除
        for col in ['stock_id', 'date_id', 'minute_id']:
            status = "✅" if col not in train_final.columns else "❌"
            print(f"{status} {col} 已从训练集移除")
        
        # 验证核心特征存在
        core_features = ['feature_13', 'feature_24', 'feature_11', 'feature_18_roll_mean_10']
        for feat in core_features:
            exists = feat in train_final.columns
            status = "✅" if exists else "❌"
            print(f"{status} 核心特征 '{feat}' {'存在' if exists else '缺失'}")
        
        # 验证无效特征已移除
        invalid_features = ['feature_13_rtn_1', 'feature_13_mom_10']
        for feat in invalid_features:
            absent = feat not in train_final.columns
            status = "✅" if absent else "❌"
            print(f"{status} 无效特征 '{feat}' {'已移除' if absent else '仍存在'}")
        
        print("\n💡 V5 设计哲学：")
        print("  • 极简主义：仅保留诊断验证有效的23个原始特征")
        print("  • 精准衍生：只生成波动率/滚动均值/交互（删除收益率/动量）")
        print("  • 自动验证：特征重要性可视化即时反馈特征质量")
        print("  • 安全第一：训练/测试完全独立，杜绝时间泄露")
        print("="*70)
        
        return train_final, test_final


# ==================== 执行入口 ====================
if __name__ == "__main__":
    DATA_DIR = Path('dataset')
    SAVE_DIR = Path('processed_v5')
    
    # 加载数据
    print("📥 加载原始数据...")
    train_raw = pd.read_csv(DATA_DIR / 'train_v2.csv')
    test_raw = pd.read_csv(DATA_DIR / 'test_v2.csv') if (DATA_DIR / 'test_v2.csv').exists() else None
    
    # 执行预处理
    preprocessor = FinancialPreprocessorV5(cold_start_day=4, save_dir=SAVE_DIR)
    train_proc, test_proc = preprocessor.fit_transform(
        train_df=train_raw,
        test_df=test_raw,
        save=True,
        visualize=True  # 启用可视化
    )
    
    print("\n🎉 预处理完成！关键产出：")
    print("  1️⃣  processed_v5/train_processed_v5.csv → 建模输入")
    print("  2️⃣  processed_v5/feature_importance.png → 特征质量验证")
    print("  3️⃣  processed_v5/feature_importance_scores.csv → 重要性分数")
    print("\n🚀 下一步行动：")
    print("  • 检查 feature_importance.png：确认 feature_13/24/11 是否在Top 5")
    print("  • 若是 → 模型性能将显著提升！")
    print("  • 若否 → 检查数据泄露或特征工程错误")