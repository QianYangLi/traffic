import torch
import torch.nn as nn
import torch.nn.functional as F


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, dilation=1):
        super().__init__()

        padding = (kernel - 1) * dilation

        self.conv = nn.Conv1d(
            in_ch, out_ch, kernel,
            padding=padding,
            dilation=dilation
        )

        self.norm = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        y = self.conv(x)
        y = y[:, :, :x.size(-1)]
        return F.relu(self.norm(y))


class TemporalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        attn = torch.softmax(
            torch.matmul(q, k.transpose(-1, -2)) / (x.size(-1) ** 0.5),
            dim=-1
        )

        return torch.matmul(attn, v)


class MultiScaleEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()

        self.tcn1 = TCNBlock(in_dim, hidden_dim, dilation=1)
        self.tcn2 = TCNBlock(in_dim, hidden_dim, dilation=2)
        self.tcn3 = TCNBlock(in_dim, hidden_dim, dilation=4)

        self.attn = TemporalAttention(hidden_dim * 3)
        self.proj = nn.Linear(hidden_dim * 3, hidden_dim)

    def forward(self, x):
        x = x.transpose(1, 2)

        f1 = self.tcn1(x)
        f2 = self.tcn2(x)
        f3 = self.tcn3(x)

        feat = torch.cat([f1, f2, f3], dim=1)
        feat = feat.transpose(1, 2)

        feat = self.attn(feat)

        return self.proj(feat)