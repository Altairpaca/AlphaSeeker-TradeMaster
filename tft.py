#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Temporal Fusion Transformer 训练脚本
专为脱敏金融分钟级数据优化 | 使用 V6.4 预处理数据
"""
import os
import warnings
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from pytorch_forecasting import (
    TimeSeriesDataSet,
    TemporalFusionTransformer,
    Baseline,
    QuantileLoss,
)
from pytorch_forecasting.data import GroupNormalizer, MultiNormalizer
from pytorch_forecasting.metrics import MAE, MultiLoss
# from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
import pickle

# 忽略烦人警告（生产环境可移除）
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class TradeMasterTFTTrainer:
    def __init__(
        self,
        train_path="processed_v6_4/train_processed_v6_4.csv",
        test_path="processed_v6_4/test_processed_v6_4.csv",
        model_dir="models/tft_v6_4",
        random_seed=42,
    ):
        self.train_path = train_path
        self.test_path = test_path
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        pl.seed_everything(random_seed)
        torch.set_float32_matmul_precision('medium')  # 速度/精度平衡
        
        # 三目标加权（短期更重要）
        self.target_weights = [0.6, 0.3, 0.1]  # short, medium, long
        
    def load_and_prepare_data(self):
        """加载 V6.4 预处理数据 + 生成时间索引"""
        print("="*70)
        print("📦 加载 V6.4 预处理数据...")
        print("="*70)
        
        train = pd.read_csv(self.train_path)
        test = pd.read_csv(self.test_path)
        
        # 验证关键列存在
        required_cols = ['date_id', 'minute_id', 'target_short', 'target_medium', 'target_long']
        for col in required_cols:
            assert col in train.columns, f"训练集缺少关键列: {col}"
            if col not in ['target_short', 'target_medium', 'target_long']:
                assert col in test.columns, f"测试集缺少关键列: {col}"
        
        # 生成时间索引（每个交易日独立序列）
        print("\n⏱️  生成时间索引 (time_idx)...")
        for df in [train, test]:
            df["time_idx"] = df.groupby("date_id").cumcount()
            df["date_id"] = df["date_id"].astype(str)  # TFT 要求 group_ids 为字符串
        
        # 特征列筛选（排除目标/ID/时间列）
        exclude_cols = {
            'id', 'date_id', 'minute_id', 'time_idx',
            'target_short', 'target_medium', 'target_long',
            'stock_id'  # 如果存在
        }
        self.feature_cols = [c for c in train.columns if c not in exclude_cols]
        
        print(f"✅ 训练集: {len(train):,} 条 | 测试集: {len(test):,} 条")
        print(f"✅ 特征数量: {len(self.feature_cols)}")
        print(f"✅ Top 5 特征: {self.feature_cols[:5]}")
        
        return train, test
    
    def create_dataset(self, train_df, validation_size=0.1):
        """创建 TFT 专用时间序列数据集"""
        print("\n" + "="*70)
        print("⚙️  创建时间序列数据集 (TimeSeriesDataSet)...")
        print("="*70)
        
        # 按 date_id 分组，取最后 validation_size 比例作为验证集
        unique_dates = sorted(train_df["date_id"].unique())
        val_split_idx = int(len(unique_dates) * (1 - validation_size))
        train_dates = unique_dates[:val_split_idx]
        val_dates = unique_dates[val_split_idx:]
        
        train_subset = train_df[train_df["date_id"].isin(train_dates)].copy()
        val_subset = train_df[train_df["date_id"].isin(val_dates)].copy()
        
        # 关键：minute_sin/cos 作为 known_reals（预测时可计算）
        time_varying_known = ["minute_sin", "minute_cos"]
        time_varying_unknown = [c for c in self.feature_cols if c not in time_varying_known]
        
        # 创建训练集
        self.training = TimeSeriesDataSet(
            train_subset,
            time_idx="time_idx",
            target=["target_short", "target_medium", "target_long"],
            group_ids=["date_id"],
            min_encoder_length=10,  # 至少用10分钟历史
            max_encoder_length=60,  # 最多用60分钟历史（覆盖1小时）
            min_prediction_length=1,
            max_prediction_length=1,
            static_categoricals=[],
            static_reals=[],
            time_varying_known_categoricals=[],
            time_varying_known_reals=time_varying_known,
            time_varying_unknown_reals=time_varying_unknown,
            target_normalizer=MultiNormalizer([
                GroupNormalizer(groups=["date_id"]),  # for target_short
                GroupNormalizer(groups=["date_id"]),  # for target_medium
                GroupNormalizer(groups=["date_id"]),  # for target_long
            ]),  # 按交易日归一化
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=False,
            lags={},
        )
        
        # 创建验证集（使用相同参数）
        self.validation = TimeSeriesDataSet.from_dataset(
            self.training, 
            val_subset, 
            predict=True, 
            stop_randomization=True
        )
        
        # DataLoader
        batch_size = 128  # 根据GPU内存调整
        self.train_dataloader = self.training.to_dataloader(
            train=True, batch_size=batch_size, num_workers=4
        )
        self.val_dataloader = self.validation.to_dataloader(
            train=False, batch_size=batch_size * 2, num_workers=4
        )
        
        print(f"✅ 训练样本数: {len(self.training)}")
        print(f"✅ 验证样本数: {len(self.validation)}")
        print(f"✅ Encoder 长度: {self.training.max_encoder_length}")
        print(f"✅ 特征维度: {len(self.training.reals)}")
        
        return self.training, self.validation
    
    def build_model(self):
        """构建 TFT 模型（针对金融数据优化）"""
        print("\n" + "="*70)
        print("🧠 构建 Temporal Fusion Transformer 模型...")
        print("="*70)
        
        # 自定义加权 MAE 损失（强调短期预测）
        class WeightedMAELoss(MultiLoss):
            def __init__(self, weights):
                super().__init__(metrics=[MAE()] * 3)
                self.weights = torch.tensor(weights, dtype=torch.float32)
            
            def __call__(self, y_pred, y_true, **kwargs):
                losses = []
                for i in range(3):
                    mae = torch.abs(y_pred[:, :, i] - y_true[:, :, i]).mean()
                    losses.append(mae * self.weights[i])
                return sum(losses)
        
        self.model = TemporalFusionTransformer.from_dataset(
            self.training,
            learning_rate=0.01,
            hidden_size=128,          # 增大容量捕捉微弱信号
            attention_head_size=4,
            dropout=0.15,             # 防止过拟合
            hidden_continuous_size=64,
            output_size=3,            # 三目标输出
            loss=WeightedMAELoss(self.target_weights),
            log_interval=10,
            reduce_on_plateau_patience=4,
            optimizer="adamw",
            weight_decay=1e-4,        # L2正则
        )
        
        print(f"✅ 模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"✅ 目标权重: short={self.target_weights[0]}, medium={self.target_weights[1]}, long={self.target_weights[2]}")
        
        return self.model
    
    def train(self, max_epochs=50):
        """训练模型 + 早停"""
        print("\n" + "="*70)
        print("🔥 开始训练 TFT 模型...")
        print("="*70)
        
        # 回调函数
        callbacks = [
            EarlyStopping(
                monitor="val_loss", 
                patience=8, 
                verbose=True, 
                mode="min"
            ),
            LearningRateMonitor(logging_interval="epoch"),
        ]
        
        # 训练器
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator="auto",
            devices="auto",
            gradient_clip_val=0.1,    # 防止梯度爆炸
            callbacks=callbacks,
            logger=TensorBoardLogger(save_dir=self.model_dir, name="tft_logs"),
            enable_checkpointing=True,
            log_every_n_steps=50,
            precision="16-mixed" if torch.cuda.is_available() else 32,  # 混合精度加速
        )
        
        # 训练
        trainer.fit(
            self.model,
            train_dataloaders=self.train_dataloader,
            val_dataloaders=self.val_dataloader,
        )
        
        # 保存最佳模型
        best_model_path = os.path.join(self.model_dir, "best_model.ckpt")
        trainer.save_checkpoint(best_model_path)
        print(f"\n✅ 模型已保存至: {best_model_path}")
        
        # 保存数据集配置（预测必需）
        dataset_config_path = os.path.join(self.model_dir, "dataset_config.pkl")
        with open(dataset_config_path, "wb") as f:
            pickle.dump({
                "training": self.training,
                "feature_cols": self.feature_cols,
                "target_weights": self.target_weights
            }, f)
        print(f"✅ 数据集配置已保存至: {dataset_config_path}")
        
        return trainer, best_model_path
    
    def predict_test(self, model_path=None):
        """生成测试集预测结果"""
        print("\n" + "="*70)
        print("🔮 生成测试集预测结果...")
        print("="*70)
        
        # 加载测试数据
        test = pd.read_csv(self.test_path)
        test["time_idx"] = test.groupby("date_id").cumcount()
        test["date_id"] = test["date_id"].astype(str)
        
        # 加载模型和数据集配置
        if model_path is None:
            model_path = os.path.join(self.model_dir, "best_model.ckpt")
        
        with open(os.path.join(self.model_dir, "dataset_config.pkl"), "rb") as f:
            config = pickle.load(f)
        
        # 创建测试集（使用训练时的 dataset 配置）
        test_dataset = TimeSeriesDataSet.from_dataset(
            config["training"], 
            test, 
            predict=True, 
            stop_randomization=True
        )
        test_dataloader = test_dataset.to_dataloader(train=False, batch_size=256, num_workers=4)
        
        # 加载模型
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TemporalFusionTransformer.load_from_checkpoint(model_path).to(device)
        model.eval()
        
        # 预测
        predictions = []
        with torch.no_grad():
            for x, _ in test_dataloader:
                # 移动到设备
                for k, v in x.items():
                    if isinstance(v, torch.Tensor):
                        x[k] = v.to(device)
                
                # 预测（取中位数预测）
                preds = model(x)
                if isinstance(preds, tuple):
                    preds = preds[0]  # 处理 Quantile 输出
                
                # preds shape: [batch, prediction_length=1, n_targets=3]
                preds = preds[:, 0, :].cpu().numpy()  # 取第一个预测步
                predictions.append(preds)
        
        # 合并预测结果
        predictions = np.vstack(predictions)
        print(f"✅ 预测完成! 形状: {predictions.shape}")
        
        # 创建提交文件
        submission = pd.DataFrame({
            "id": test["id"],
            "target_short": predictions[:, 0],
            "target_medium": predictions[:, 1],
            "target_long": predictions[:, 2]
        })
        
        # 保存
        sub_path = os.path.join(self.model_dir, "submission_tft_v6_4.csv")
        submission.to_csv(sub_path, index=False)
        print(f"✅ 提交文件已保存至: {sub_path}")
        print(f"📊 预测统计: short_mean={submission['target_short'].mean():.6f}, "
              f"short_std={submission['target_short'].std():.6f}")
        
        return submission
    
    def run_full_pipeline(self):
        """端到端执行：数据 → 训练 → 预测"""
        # 1. 加载数据
        train_df, test_df = self.load_and_prepare_data()
        
        # 2. 创建数据集
        self.create_dataset(train_df)
        
        # 3. 构建模型
        self.build_model()
        
        # 4. 训练
        trainer, model_path = self.train(max_epochs=50)
        
        # 5. 预测
        submission = self.predict_test(model_path)
        
        print("\n" + "="*70)
        print("🎉 TFT 训练全流程完成!")
        print(f"📁 模型路径: {self.model_dir}")
        print(f"📁 提交文件: {os.path.join(self.model_dir, 'submission_tft_v6_4.csv')}")
        print("="*70)
        
        return submission

# ==================== 主执行流程 ====================
if __name__ == "__main__":
    # 配置路径（根据你的实际路径修改）
    TRAINER = TradeMasterTFTTrainer(
        train_path="processed_v6_4/train_processed_v6_4.csv",
        test_path="processed_v6_4/test_processed_v6_4.csv",
        model_dir="models/tft_v6_4",
        random_seed=42,
    )
    
    # 执行全流程
    submission = TRAINER.run_full_pipeline()
    
    # 可选：快速验证预测分布
    print("\n🔍 预测分布验证:")
    print(submission[["target_short", "target_medium", "target_long"]].describe())