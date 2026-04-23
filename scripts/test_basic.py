import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
from torch.utils.data import DataLoader

from data.dataset import (
    load_dataset,
    split_dataset,
    normalize,
    TrafficDataset,
)
from model.baselines import (
    LSTM_TM,
    BiLSTM_TM,
    GRU_TM,
    BiGRU_TM,
)
from configs.config import HISTORY_LEN, PRED_LEN, BATCH_SIZE


EPS = 1e-8


def mae(pred, true):
    return torch.mean(torch.abs(pred - true))


def mse(pred, true):
    return torch.mean((pred - true) ** 2)


def rmse(pred, true):
    return torch.sqrt(torch.mean((pred - true) ** 2))


def mape(pred, true, eps=EPS):
    return torch.mean(torch.abs((pred - true) / (torch.abs(true) + eps))) * 100


def build_model(model_name, input_dim, hidden_dim=128):
    name = model_name.lower()

    if name == "lstm":
        return LSTM_TM(input_dim=input_dim, hidden_dim=hidden_dim)
    elif name == "bilstm":
        return BiLSTM_TM(input_dim=input_dim, hidden_dim=hidden_dim)
    elif name == "gru":
        return GRU_TM(input_dim=input_dim, hidden_dim=hidden_dim)
    elif name == "bigru":
        return BiGRU_TM(input_dim=input_dim, hidden_dim=hidden_dim)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 这里切换模型: "lstm" / "bilstm" / "gru" / "bigru"
    model_name = "bigru"

    data = load_dataset("data/traffic_martixes")

    train_data, val_data, test_data = split_dataset(data)

    train_data, stat_a, stat_b = normalize(train_data)
    test_data, _, _ = normalize(test_data, stat_a, stat_b)

    test_set = TrafficDataset(test_data, HISTORY_LEN, PRED_LEN)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    input_dim = data.shape[1] * data.shape[2]   # 529

    model = build_model(model_name, input_dim=input_dim, hidden_dim=128).to(device)

    model_path = f"best_model_xml_{model_name}.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Cannot find {model_path}. Please run train_basic_rnn.py first.")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    preds_all = []
    trues_all = []
    x_last_all = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)                # [B, T, 529]
            y = y.to(device).squeeze(1)     # [B, 529]

            pred = model(x)                 # [B, 529]

            preds_all.append(pred.cpu())
            trues_all.append(y.cpu())
            x_last_all.append(x[:, -1, :].cpu())

    preds_all = torch.cat(preds_all, dim=0)
    trues_all = torch.cat(trues_all, dim=0)
    x_last_all = torch.cat(x_last_all, dim=0)

    print(f"===== {model_name.upper()} XML Test Metrics =====")
    print("MAE :", mae(preds_all, trues_all).item())
    print("MSE :", mse(preds_all, trues_all).item())
    print("RMSE:", rmse(preds_all, trues_all).item())
    print("MAPE:", mape(preds_all, trues_all).item())

    print("\n===== Baseline Check =====")
    print("Copy-last-step MAE :", mae(x_last_all, trues_all).item())
    print("Pred vs x_last MAE :", mae(preds_all, x_last_all).item())
    print("True vs x_last MAE :", mae(trues_all, x_last_all).item())


if __name__ == "__main__":
    main()