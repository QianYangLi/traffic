import torch
import torch.nn as nn


class LSTM_TM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            dropout=0.2,
        )
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        """
        x: [B, T, D]
        return: [B, D]
        """
        h, _ = self.rnn(x)          # [B, T, H]
        y = self.fc(h[:, -1, :])    # [B, D]
        return y


class BiLSTM_TM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )
        self.fc = nn.Linear(2 * hidden_dim, input_dim)

    def forward(self, x):
        h, _ = self.rnn(x)
        y = self.fc(h[:, -1, :])
        return y


class GRU_TM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            dropout=0.2,
        )
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        h, _ = self.rnn(x)
        y = self.fc(h[:, -1, :])
        return y


class BiGRU_TM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )
        self.fc = nn.Linear(2 * hidden_dim, input_dim)

    def forward(self, x):
        h, _ = self.rnn(x)
        y = self.fc(h[:, -1, :])
        return y