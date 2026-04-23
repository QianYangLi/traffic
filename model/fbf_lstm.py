import torch
import torch.nn as nn


class FBF_LSTM(nn.Module):
    """
    Multi-variate FBF-LSTM for traffic matrix forecasting.

    Input:
        x: [B, seq_len, in_dim]

    Output:
        y_hat: [B, pre_len, in_dim]
    """
    def __init__(self, in_dim, hidden_dim, n_layer=3, seq_len=12, pre_len=1, dropout=0.5):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.pre_len = pre_len

        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=n_layer,
            batch_first=True,
            dropout=dropout if n_layer > 1 else 0.0,
        )

        # project hidden state of each history step -> feature dimension
        self.feature_proj = nn.Linear(hidden_dim, in_dim)

        # map time dimension: seq_len -> pre_len
        self.time_linear = nn.Linear(seq_len, pre_len)

    def forward(self, x):
        """
        x: [B, seq_len, in_dim]
        return: [B, pre_len, in_dim]
        """
        h, _ = self.lstm(x)                       # [B, seq_len, hidden_dim]
        h = self.feature_proj(h)                  # [B, seq_len, in_dim]

        # convert [B, seq_len, in_dim] -> [B, in_dim, seq_len]
        h = h.transpose(1, 2)                     # [B, in_dim, seq_len]

        # map time axis
        y = self.time_linear(h)                   # [B, in_dim, pre_len]

        # back to [B, pre_len, in_dim]
        y = y.transpose(1, 2)                     # [B, pre_len, in_dim]
        return y