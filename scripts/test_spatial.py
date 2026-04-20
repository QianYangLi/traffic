import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
from torch.utils.data import DataLoader

from data.dataset import (
    load_dataset,
    split_dataset,
    normalize,
    TrafficDataset
)

from model.predictor_spatiotemporal import TrafficPredictorSpatioTemporal
from configs.config import HIDDEN_DIM, HISTORY_LEN, PRED_LEN, BATCH_SIZE, SAMPLES


def mae(pred, true):
    return torch.mean(torch.abs(pred - true))


def mse(pred, true):
    return torch.mean((pred - true) ** 2)


def rmse(pred, true):
    return torch.sqrt(torch.mean((pred - true) ** 2))


def main():
    # =========================================================
    # 1. Device
    # =========================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # =========================================================
    # 2. Load G?ANT/XML dataset
    # =========================================================
    data = load_dataset("data/traffic_martixes")

    # =========================================================
    # 3. Split + normalize
    # =========================================================
    train_data, val_data, test_data = split_dataset(data)

    train_data, stat_a, stat_b = normalize(train_data)
    test_data, _, _ = normalize(test_data, stat_a, stat_b)

    # =========================================================
    # 4. Build test dataset
    # =========================================================
    test_set = TrafficDataset(test_data, HISTORY_LEN, PRED_LEN)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    # =========================================================
    # 5. Build model
    # =========================================================
    num_nodes = data.shape[1]
    dim = data.shape[1] * data.shape[2]

    A_static = torch.eye(num_nodes)

    model = TrafficPredictorSpatioTemporal(
        in_dim=dim,
        hidden_dim=HIDDEN_DIM,
        out_dim=dim,
        num_nodes=num_nodes,
        A_static=A_static
    ).to(device)

    model_path = "best_model_xml_spatial.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Cannot find {model_path}. Please run train_spatial.py first."
        )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # =========================================================
    # 6. Test loop
    # =========================================================
    preds_all = []
    trues_all = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)                 # [B, hist_len, 529]
            y = y.to(device)                 # [B, 1, 529]

            # 多次采样后取均值作为最终预测
            samples = model.predict(x, samples=SAMPLES)     # list of [B, 529]
            pred = torch.stack(samples, dim=0).mean(dim=0)  # [B, 529]

            preds_all.append(pred.cpu())
            trues_all.append(y.squeeze(1).cpu())

    preds_all = torch.cat(preds_all, dim=0)
    trues_all = torch.cat(trues_all, dim=0)

    # =========================================================
    # 7. Metrics
    # =========================================================
    print("===== Spatial-Temporal XML Test Metrics =====")
    print("MAE :", mae(preds_all, trues_all).item())
    print("MSE :", mse(preds_all, trues_all).item())
    print("RMSE:", rmse(preds_all, trues_all).item())


if __name__ == "__main__":
    main()