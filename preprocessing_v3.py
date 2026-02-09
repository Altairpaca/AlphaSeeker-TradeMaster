# financial_preprocessor_v4.py
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class FinancialPreprocessorV4:
    """
    专为单股票+时间不连续场景设计：
    ✅ 训练/测试集完全独立处理（杜绝gap导致的泄露）
    ✅ 精准修复feature_11系统性缺失（分钟237-238）
    ✅ 仅保留高信号特征工程（收益率/波动率为核心）
    ✅ 自动移除无信息量ID列（stock_id/date_id/minute_id）
    ✅ 测试集目标列安全处理（预测时自动清理）
    """
    
    def __init__(self, cold_start_day=4, save_dir='processed_v4'):
        self.cold_start_day = cold_start_day
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.fill_values = {}  # 仅训练集冷启动填充值
        self.feature_cols = [f'feature_{i}' for i in range(1, 31)]
        self.price_proxy_features = ['feature_13', 'feature_14', 'feature_24']  # 诊断验证高相关
        self.rolling_features = ['feature_18', 'feature_21', 'feature_25']      # 诊断验证有效
    
    def _load_and_prepare(self, df, is_train=True):
        """基础清理：替换inf，确保时间排序，标记来源"""
        df = df.copy()
        # 安全替换inf
        df[self.feature_cols] = df[self.feature_cols].replace([np.inf, -np.inf], np.nan)
        # 严格按时间排序（关键！）
        if 'date_id' in df.columns and 'minute_id' in df.columns:
            df = df.sort_values(['date_id', 'minute_id']).reset_index(drop=True)
        df['_is_train'] = 1 if is_train else 0
        return df
    
    def _handle_cold_start_missing(self, df):
        """
        仅处理训练集冷启动缺失（前cold_start_day天）
        测试集无此问题（date_id从582开始，数据完整）
        """
        df_processed = df.copy()
        cold_start_mask = (df_processed['_is_train'] == 1) & (df_processed['date_id'] < self.cold_start_day)
        
        if cold_start_mask.any():
            print(f"❄️  处理训练集冷启动缺失: 前 {self.cold_start_day} 天 (共 {cold_start_mask.sum()} 条)")
            for feat in self.feature_cols:
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
        
        # 保留缺失指示（测试集也需要！）
        df_processed['f11_sys_missing'] = systemic_mask.astype(int)
        df_processed['f11_rand_missing'] = (original_missing & ~systemic_mask).astype(int)
        print(f"📌 创建缺失指示: f11_sys_missing={systemic_mask.sum()}, f11_rand_missing={(original_missing & ~systemic_mask).sum()}")
        return df_processed
    
    def _handle_other_missing(self, df):
        """健壮性处理其他特征随机缺失（仅cold_start_day后）"""
        df_processed = df.copy()
        other_features = [f for f in self.feature_cols if f != 'feature_11']
        
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
    
    def _create_return_features(self, df):
        """核心！仅对高信号特征构造收益率/波动率（单股票场景：stock_id全0，分组安全）"""
        df_engineered = df.copy()
        for feat in self.price_proxy_features:
            if feat not in df_engineered.columns:
                continue
            # 收益率（分钟级变化）
            df_engineered[f'{feat}_rtn_1'] = df_engineered.groupby('stock_id')[feat].pct_change(1).fillna(0)
            df_engineered[f'{feat}_rtn_5'] = df_engineered.groupby('stock_id')[feat].pct_change(5).fillna(0)
            # 波动率
            df_engineered[f'{feat}_vol_10'] = df_engineered.groupby('stock_id')[f'{feat}_rtn_1'].transform(
                lambda x: x.rolling(10, min_periods=1).std()
            ).fillna(0)
            # 动量
            df_engineered[f'{feat}_mom_10'] = (
                df_engineered[feat] - 
                df_engineered.groupby('stock_id')[feat].transform(
                    lambda x: x.rolling(10, min_periods=1).mean()
                )
            ).fillna(0)
        print(f"📈 创建收益率特征: 基于 {self.price_proxy_features}")
        return df_engineered
    
    def _create_time_features(self, df):
        """轻量级时间特征（替代原始minute_id）"""
        df_engineered = df.copy()
        df_engineered['is_near_close'] = (df_engineered['minute_id'] >= 230).astype(int)  # 收盘前10分钟
        df_engineered['trading_progress'] = df_engineered['minute_id'] / 239.0  # 归一化进度
        return df_engineered
    
    def _create_rolling_features(self, df):
        """仅保留诊断验证有效的滚动特征"""
        df_engineered = df.copy()
        for feat in self.rolling_features:
            if feat in df_engineered.columns:
                df_engineered[f'{feat}_roll_mean_10'] = df_engineered.groupby('stock_id')[feat].transform(
                    lambda x: x.rolling(10, min_periods=1).mean()
                ).fillna(0)
        print(f"🌀 创建滚动特征: {self.rolling_features}")
        return df_engineered
    
    def _create_interactions(self, df):
        """仅保留诊断验证有效的交互特征"""
        df_engineered = df.copy()
        if 'feature_11' in df_engineered.columns and 'feature_18' in df_engineered.columns:
            df_engineered['f11_f18'] = df_engineered['feature_11'] * df_engineered['feature_18']
        if 'feature_21' in df_engineered.columns and 'feature_25' in df_engineered.columns:
            df_engineered['f21_f25'] = df_engineered['feature_21'] * df_engineered['feature_25']
        print("🔀 创建交互特征: f11_f18, f21_f25")
        return df_engineered
    
    def _final_sanity_check(self, df):
        """最终检查：填充剩余缺失，保留_is_train供后续拆分"""
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
        """
        移除无信息量列：
        - stock_id: 全0，无区分度
        - date_id: 训练/测试不连续，保留会导致过拟合
        - minute_id: 已转化为时间特征
        - _is_train: 临时标记
        - 测试集额外移除目标列（预测时不需要）
        """
        cols_to_drop = ['stock_id', 'date_id', 'minute_id', '_is_train']
        if is_test:
            cols_to_drop += ['target_short', 'target_medium', 'target_long']
        return df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
    
    def fit_transform(self, train_df, test_df=None, save=True):
        """
        核心改进：训练集与测试集完全独立处理（应对时间不连续）
        """
        print("="*70)
        print("🚀 金融数据预处理 V4 | 单股票 + 时间不连续场景专用")
        print("="*70)
        
        # =============== 1. 分别加载与准备 ===============
        print("\n[1/2] 处理训练集")
        train_proc = self._load_and_prepare(train_df, is_train=True)
        print(f"   📊 训练集: {len(train_proc)} 条 | date_id范围: [{train_proc['date_id'].min()}, {train_proc['date_id'].max()}]")
        
        if test_df is not None:
            # 确保测试集有目标列占位（避免后续操作报错，但最终会删除）
            for tgt in ['target_short', 'target_medium', 'target_long']:
                if tgt not in test_df.columns:
                    test_df[tgt] = np.nan
            test_proc = self._load_and_prepare(test_df, is_train=False)
            print(f"\n[2/2] 处理测试集")
            print(f"   📊 测试集: {len(test_proc)} 条 | date_id范围: [{test_proc['date_id'].min()}, {test_proc['date_id'].max()}]")
            print(f"   ⚠️  时间不连续: 训练集结束于 date_id={train_proc['date_id'].max()}, 测试集开始于 date_id={test_proc['date_id'].min()}")
        else:
            test_proc = None
        
        # =============== 2. 独立处理训练集 ===============
        print("\n" + "-"*70)
        print("🔧 训练集处理流程")
        print("-"*70)
        train_proc = self._handle_cold_start_missing(train_proc)      # 仅训练集有冷启动
        train_proc = self._handle_feature_11_missing(train_proc)
        train_proc = self._handle_other_missing(train_proc)
        train_proc = self._create_return_features(train_proc)         # 核心！
        train_proc = self._create_time_features(train_proc)
        train_proc = self._create_rolling_features(train_proc)
        train_proc = self._create_interactions(train_proc)
        train_proc = self._final_sanity_check(train_proc)
        train_final = self._remove_unnecessary_columns(train_proc, is_test=False)
        
        # =============== 3. 独立处理测试集 ===============
        if test_proc is not None:
            print("\n" + "-"*70)
            print("🔧 测试集处理流程（完全独立，不使用训练集任何数据）")
            print("-"*70)
            # 跳过冷启动处理（测试集无此问题，函数内部有判断）
            test_proc = self._handle_cold_start_missing(test_proc)    # 实际无操作
            test_proc = self._handle_feature_11_missing(test_proc)    # 修复测试集系统性缺失
            test_proc = self._handle_other_missing(test_proc)
            test_proc = self._create_return_features(test_proc)       # 仅用测试集内部历史
            test_proc = self._create_time_features(test_proc)
            test_proc = self._create_rolling_features(test_proc)
            test_proc = self._create_interactions(test_proc)
            test_proc = self._final_sanity_check(test_proc)
            test_final = self._remove_unnecessary_columns(test_proc, is_test=True)
        else:
            test_final = None
        
        # =============== 4. 保存与验证 ===============
        if save:
            # 保存训练集（含目标列）
            train_path = self.save_dir / 'train_processed_v4.csv'
            train_final.to_csv(train_path, index=False)
            print(f"\n✅ 训练集保存: {train_path} | 形状: {train_final.shape}")
            
            # 保存测试集（无目标列，符合预测要求）
            if test_final is not None:
                test_path = self.save_dir / 'test_processed_v4.csv'
                test_final.to_csv(test_path, index=False)
                print(f"✅ 测试集保存: {test_path} | 形状: {test_final.shape} (已移除目标列)")
            
            # 保存特征列表（训练集特征列，不含目标）
            feature_cols = [c for c in train_final.columns if c not in ['id', 'target_short', 'target_medium', 'target_long']]
            pd.Series(sorted(feature_cols)).to_csv(self.save_dir / 'feature_list_v4.txt', index=False, header=False)
            print(f"📋 特征列表保存: {len(feature_cols)} 个特征 (已移除stock_id/date_id/minute_id)")
        
        # =============== 5. 关键验证 ===============
        print("\n" + "="*70)
        print("🔍 处理验证")
        print("="*70)
        # 验证ID列已移除
        for col in ['stock_id', 'date_id', 'minute_id']:
            if col in train_final.columns:
                print(f"❌ 错误: {col} 仍在训练集中！")
            else:
                print(f"✅ {col} 已从训练集移除")
        
        # 验证feature_11修复
        if 'feature_11' in train_final.columns:
            print(f"✅ 训练集 feature_11 无缺失: {train_final['feature_11'].isna().sum() == 0}")
        if test_final is not None and 'feature_11' in test_final.columns:
            print(f"✅ 测试集 feature_11 无缺失: {test_final['feature_11'].isna().sum() == 0}")
        
        # 验证核心特征存在
        core_features = ['feature_13_rtn_1', 'feature_14_vol_10', 'is_near_close']
        for feat in core_features:
            exists = feat in train_final.columns
            status = "✅" if exists else "❌"
            print(f"{status} 核心特征 '{feat}' {'存在' if exists else '缺失'}")
        
        # 验证测试集无目标列
        if test_final is not None:
            target_in_test = any(tgt in test_final.columns for tgt in ['target_short', 'target_medium', 'target_long'])
            print(f"{'✅' if not target_in_test else '❌'} 测试集目标列已移除: {not target_in_test}")
        
        print("\n💡 为什么这样设计？")
        print("  • 独立处理: 训练/测试时间不连续（580→582），合并会导致滚动特征跨gap泄露")
        print("  • 移除ID列: stock_id全0无信息；date_id不连续且无日历信息；minute_id已转化为特征")
        print("  • 保留时间特征: is_near_close/trading_progress 捕捉日内模式")
        print("  • 测试集安全: 目标列自动移除，符合预测流程")
        print("="*70)
        
        return train_final, test_final


# ==================== 执行入口 ====================
if __name__ == "__main__":
    DATA_DIR = Path('dataset')
    SAVE_DIR = Path('processed_v4')
    
    # 加载数据
    print("📥 加载原始数据...")
    train_raw = pd.read_csv(DATA_DIR / 'train_v2.csv')
    test_raw = pd.read_csv(DATA_DIR / 'test_v2.csv') if (DATA_DIR / 'test_v2.csv').exists() else None
    
    # 执行预处理
    preprocessor = FinancialPreprocessorV4(cold_start_day=4, save_dir=SAVE_DIR)
    train_proc, test_proc = preprocessor.fit_transform(
        train_df=train_raw,
        test_df=test_raw,
        save=True
    )
    
    print("\n🎉 预处理完成！下一步行动：")
    print("  1️⃣  运行 feature_diagnostic.py 评估新特征（使用 train_processed_v4.csv）")
    print("  2️⃣  重点关注: feature_13_rtn_1 与 target_short 的 lag-1 相关性")
    print("  3️⃣  若 |corr| > 0.05 → 模型性能将显著提升")
    print("  4️⃣  若仍低 → 接受现实：target_long 预测0，聚焦 short/medium")
    print("\n✨ 核心优势：无信息泄露 + 无冗余特征 + 严格适配单股票场景")