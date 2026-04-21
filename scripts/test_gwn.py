import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import load_dataset, split_dataset, normalize, TrafficDataset
from model.gwn import GWNet
from configs.config import HISTORY_LEN, PRED_LEN, BATCH_SIZE


def build_flow_correlation_graph(train_data, threshold=0.3):
    T, N, _ = train_data.shape
    flow_data = train_data.reshape(T, -1)

    with np.errstate(divide="ignore", invalid="ignore"):
        A = np.corrcoef(flow_data.T)

    A = np.nan_to_num(A)
    A = (A + 1.0) / 2.0
    A[A < threshold] = 0.0
    return A.astype(np.float32)


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

    num_flows = data.shape[1] * data.shape[2]

    A_np = build_flow_correlation_graph(train_data, threshold=0.3)
    A_static = torch.tensor(A_np, dtype=torch.float32, device=device)

    model = GWNet(
        device=device,
        num_nodes=num_flows,
        dropout=0.3,
        supports=[A_static],
        do_graph_conv=True,
        addaptadj=True,
        aptinit=None,
        in_dim=1,
        out_seq_len=PRED_LEN,
        residual_channels=32,
        dilation_channels=32,
        skip_channels=256,
        end_channels=512,
        kernel_size=2,
        blocks=4,
        layers=2,
        stride=2,
        apt_size=10,
        verbose=0,
    ).to(device)

    model_path = "best_model_xml_gwn.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Cannot find {model_path}. Please run train_gwn.py first.")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    preds_all = []
    trues_all = []
    x_last_all = []

    with torch.no_grad():
        for x, y in test_loader:
            x_in = x.unsqueeze(-1).to(device)      # [B, T, 529, 1]
            y = y.to(device)                       # [B, 1, 529]

            pred = model(x_in)                     # [B, 1, 529, T']
            pred = pred[..., -1]                   # [B, 1, 529]

            preds_all.append(pred.cpu())
            trues_all.append(y.cpu())
            x_last_all.append(x[:, -1, :].unsqueeze(1).cpu())

    preds_all = torch.cat(preds_all, dim=0)
    trues_all = torch.cat(trues_all, dim=0)
    x_last_all = torch.cat(x_last_all, dim=0)

    print("===== GWNet XML Test Metrics =====")
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