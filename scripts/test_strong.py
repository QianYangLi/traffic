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
    build_correlation_graph,
)
from model.predictor_st_strong import TrafficPredictorSTStrong
from configs.config import HISTORY_LEN, PRED_LEN, BATCH_SIZE, HIDDEN_DIM


def mae(pred, true):
    return torch.mean(torch.abs(pred - true))


def mse(pred, true):
    return torch.mean((pred - true) ** 2)


def rmse(pred, true):
    return torch.sqrt(torch.mean((pred - true) ** 2))


def mape(pred, true, eps=1e-8):
    return torch.mean(torch.abs((pred - true) / (torch.abs(true) + eps))) * 100


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    data = load_dataset("data/traffic_martixes")

    train_data, val_data, test_data = split_dataset(data)

    train_data, stat_a, stat_b = normalize(train_data)
    test_data, _, _ = normalize(test_data, stat_a, stat_b)

    test_set = TrafficDataset(test_data, HISTORY_LEN, PRED_LEN)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    num_nodes = data.shape[1]
    dim = data.shape[1] * data.shape[2]

    A_np = build_correlation_graph(train_data, method="row", threshold=0.3)
    A_static = torch.tensor(A_np, dtype=torch.float32, device=device)

    model = TrafficPredictorSTStrong(
        in_dim=dim,
        hidden_dim=HIDDEN_DIM,
        out_dim=dim,
        num_nodes=num_nodes,
        history_len=HISTORY_LEN,
        A_static=A_static,
        num_blocks=6,
        dropout=0.1,
    ).to(device)

    model_path = "best_model_xml_strong.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Cannot find {model_path}. Please run train_strong.py first.")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    preds_all = []
    trues_all = []
    x_last_all = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device).squeeze(1)

            pred = model(x)

            preds_all.append(pred.cpu())
            trues_all.append(y.cpu())
            x_last_all.append(x[:, -1, :].cpu())

    preds_all = torch.cat(preds_all, dim=0)
    trues_all = torch.cat(trues_all, dim=0)
    x_last_all = torch.cat(x_last_all, dim=0)

    print("===== Strong ST Discriminative Test Metrics =====")
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