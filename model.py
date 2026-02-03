import math
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_INPUT_SIZE = 99


class WeightedMAELoss(nn.Module):
    def __init__(self, weights=None, device='cuda', dtype=torch.float32):
        super().__init__()
        # 根据比赛重要性设置权重，如[0.3, 0.3, 0.4]表示更重视长期预测
        self.weights = torch.tensor(weights if weights else [
                                    1.0, 1.0, 1.0]).to(device, dtype=dtype)
        self.dtype = dtype

    def forward(self, pred, target):
        # pred和target形状: (batch_size, seq_len, 3)
        abs_errors = torch.abs(pred - target)  # 计算每个目标的绝对误差
        # 按目标维度加权平均
        weighted_errors = abs_errors @ self.weights
        return weighted_errors.mean(dtype=self.dtype)


class LSTMBaseLine(nn.Module):
    def __init__(self, input_size=DEFAULT_INPUT_SIZE, target_num=3, hidden_dim=48, num_layers=1, dropout=0.2):
        super(LSTMBaseLine, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_dim,
                            num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, target_num)

    def forward(self, x):
        out, _ = self.lstm(x)
        # [batch_size, seq_len, hidden_dim]
        out = self.fc(out)
        return out


class GRUBaseLine(nn.Module):
    def __init__(self, input_size=DEFAULT_INPUT_SIZE, target_num=3, hidden_dim=48, num_layers=1, dropout=0.2):
        super(GRUBaseLine, self).__init__()
        self.gru = nn.GRU(input_size, hidden_dim, num_layers,
                          batch_first=True, dropout=dropout)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),  # GRU输出 + 注意力上下文
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, target_num)
        )

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.output_layer(out)
        return out

class UltraLightGRUModel(nn.Module):
    """
    超轻量级模型：因果卷积 + GRU + 简化注意力
    内存消耗极低
    修复了卷积padding问题，确保输入输出序列长度一致
    """

    def __init__(self, input_dim=DEFAULT_INPUT_SIZE, hidden_dim=48, dropout=0.2):
        super().__init__()

        # 1. 因果卷积提取局部特征
        # 公式：输出长度 = 输入长度 + 2*padding - (kernel_size-1)
        # 当padding=1, kernel_size=3时：输出长度 = 输入长度 + 2 * 1 - 2 = 输入长度
        self.causal_conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3,
                      padding=2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3,
                      padding=2),
            nn.ReLU(),
        )

        # 2. GRU时序建模
        self.gru = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=0
        )

        # 3. 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),  # GRU输出 + 注意力上下文
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3)
        )

    def forward(self, x):
        """
        处理输入形状：
        - 如果有批次维度: (batch, seq_len, input_dim)
        - 如果无批次维度: (seq_len, input_dim)
        都转换为有批次维度处理
        """
        # 记录原始维度
        has_batch_dim = len(x.shape) == 3

        if not has_batch_dim:
            # 添加批次维度: (seq_len, input_dim) -> (1, seq_len, input_dim)
            x = x.unsqueeze(0)

        # batch_size, seq_len, input_dim = x.shape
        seq_len = x.shape[1]

        # 1. 因果卷积
        x_conv = x.transpose(1, 2)  # (batch, input_dim, seq_len)
        conv_out = self.causal_conv(x_conv)  # (batch, hidden_dim, seq_len)
        conv_out = conv_out.transpose(1, 2)[:, :seq_len, :]  # (batch, seq_len, hidden_dim)
        
        # 2. GRU处理
        gru_out, _ = self.gru(conv_out)  # (batch, seq_len, hidden_dim)

        # 5. 输出
        output = self.output_layer(gru_out)  # (batch, seq_len, 3)

        # 如果输入没有批次维度，移除批次维度
        if not has_batch_dim:
            output = output.squeeze(0)  # (seq_len, 3)

        return output


class CausalLightGRUModel(nn.Module):
    """
    完全因果的轻量级模型
    确保不包含任何未来信息
    """

    def __init__(self, input_dim=DEFAULT_INPUT_SIZE, hidden_dim=48, dropout=0.2):
        super().__init__()

        # 1. 因果卷积提取局部特征
        # 使用膨胀因果卷积，确保只看到过去
        self.conv1 = nn.Conv1d(input_dim, hidden_dim,
                               kernel_size=3, padding=2, dilation=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim,
                               kernel_size=3, padding=4, dilation=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # 2. GRU时序建模（单向，保持因果）
        self.gru = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=0
        )

        # 3. 因果注意力（只看到历史）
        # 使用因果掩码的简单注意力
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

        # 4. 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3)
        )

    def _causal_attention(self, x):
        """
        实现因果注意力：位置i只能看到位置j (j <= i)
        """
        batch_size, seq_len, hidden_dim = x.shape

        # 投影
        query = self.query_proj(x)  # (batch, seq_len, hidden_dim)
        key = self.key_proj(x)      # (batch, seq_len, hidden_dim)
        value = self.value_proj(x)  # (batch, seq_len, hidden_dim)

        # 计算注意力分数
        scores = torch.bmm(query, key.transpose(1, 2)) / (hidden_dim ** 0.5)

        # 应用因果掩码
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1).to(x.device)
        scores = scores.masked_fill(mask, float('-inf'))

        # Softmax
        attention_weights = F.softmax(scores, dim=-1)

        # 加权求和
        attended = torch.bmm(attention_weights, value)

        return attended

    def forward(self, x):
        # 处理批次维度
        has_batch_dim = len(x.shape) == 3

        if not has_batch_dim:
            x = x.unsqueeze(0)

        batch_size, seq_len, input_dim = x.shape

        # 1. 因果卷积
        x_conv = x.transpose(1, 2)  # (batch, input_dim, seq_len)

        # 第一层因果卷积
        conv1_out = self.conv1(x_conv)  # 输出长度可能增加，需要裁剪
        conv1_out = conv1_out[:, :, :seq_len]  # 保持长度
        conv1_out = self.relu(conv1_out)
        conv1_out = self.dropout(conv1_out)

        # 第二层因果卷积
        conv_out = self.conv2(conv1_out)
        conv_out = conv_out[:, :, :seq_len]  # 保持长度
        conv_out = self.relu(conv_out)

        conv_out = conv_out.transpose(1, 2)  # (batch, seq_len, hidden_dim)

        # 2. GRU处理
        gru_out, _ = self.gru(conv_out)  # (batch, seq_len, hidden_dim)

        # 3. 因果注意力
        attn_out = self._causal_attention(gru_out)

        # 4. 残差连接
        combined = gru_out + attn_out  # 残差连接

        # 5. 输出
        output = self.output_layer(combined)  # (batch, seq_len, 3)

        if not has_batch_dim:
            output = output.squeeze(0)

        return output
