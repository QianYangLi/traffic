import torch
import torch.nn as nn
import torch.nn.functional as F


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, dilation=1):
        super().__init__()

        padding = (kernel - 1) * dilation

        self.conv = nn.Conv1d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel,
            padding=padding,
            dilation=dilation
        )
        self.norm = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        """
        x: [B, C, T]
        """
        y = self.conv(x)
        y = y[:, :, :x.size(-1)]   # causal trim
        y = self.norm(y)
        y = F.relu(y)
        return y


class TemporalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

    def forward(self, x):
        """
        x: [B, T, D]
        """
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        attn = torch.matmul(q, k.transpose(-1, -2)) / (x.size(-1) ** 0.5)
        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        return out


class MultiScaleEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()

        # 输入 x 是 [B, T, D]
        # 转置后给 Conv1d: [B, D, T]
        self.tcn1 = TCNBlock(in_dim, hidden_dim, kernel=3, dilation=1)
        self.tcn2 = TCNBlock(in_dim, hidden_dim, kernel=3, dilation=2)
        self.tcn3 = TCNBlock(in_dim, hidden_dim, kernel=3, dilation=4)

        self.attn = TemporalAttention(hidden_dim * 3)
        self.proj = nn.Linear(hidden_dim * 3, hidden_dim)

    def forward(self, x):
        """
        x: [B, T, D]
        return: [B, T, H]
        """
        x = x.transpose(1, 2)  # [B, D, T]

        f1 = self.tcn1(x)      # [B, H, T]
        f2 = self.tcn2(x)
        f3 = self.tcn3(x)

        feat = torch.cat([f1, f2, f3], dim=1)   # [B, 3H, T]
        feat = feat.transpose(1, 2)             # [B, T, 3H]

        feat = self.attn(feat)                  # [B, T, 3H]
        feat = self.proj(feat)                  # [B, T, H]

        return feat