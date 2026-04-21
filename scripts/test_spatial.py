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
    # 2. Load dataset
    # =========================================================
    data = load_dataset("data/traffic_martixes")

    # =========================================================
    # 3. Split + normalize
    # =========================================================
    train_data, val_data, test_data = split_dataset(data)

    train_data, stat_a, stat_b = normalize(train_data)
    test_data, _, _ = normalize(test_data, stat_a, stat_b)

    # =========================================================
    # 4. Dataset
    # =========================================================
    test_set = TrafficDataset(test_data, HISTORY_LEN, PRED_LEN)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    # =========================================================
    # 5. Model
    # =========================================================
    num_nodes = data.shape[1]
    dim = data.shape[1] * data.shape[2]

    from data.dataset import build_correlation_graph

# 使用训练数据构图（非常重要）
    A_np = build_correlation_graph(train_data, method="row", threshold=0.3)

    A_static = torch.tensor(A_np, dtype=torch.float32).to(device)

    model = TrafficPredictorSpatioTemporal(
        in_dim=dim,
        hidden_dim=HIDDEN_DIM,
        out_dim=dim,
        num_nodes=num_nodes,
        A_static=A_static,
        aux_weight=0.1
    ).to(device)

    model_path = "best_model_xml_spatial.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Cannot find {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # =========================================================
    # 6. Inference
    # =========================================================
    preds_mean_all = []
    preds_median_all = []
    preds_aux_all = []
    trues_all = []
    x_last_all = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            x_last = x[:, -1, :]
            y_true = y.squeeze(1)

            # ===== Diffusion =====
            samples = model.predict(x, samples=SAMPLES)
            samples = torch.stack(samples, dim=0)  # [S, B, D]

            delta_mean = samples.mean(dim=0)
            delta_median = samples.median(dim=0).values

            pred_mean = x_last + delta_mean
            pred_median = x_last + delta_median

            # ===== Auxiliary Head =====
            pred_aux = model.predict_aux(x)

            # ===== Collect =====
            preds_mean_all.append(pred_mean.cpu())
            preds_median_all.append(pred_median.cpu())
            preds_aux_all.append(pred_aux.cpu())
            trues_all.append(y_true.cpu())
            x_last_all.append(x_last.cpu())

    # concat
    preds_mean_all = torch.cat(preds_mean_all, dim=0)
    preds_median_all = torch.cat(preds_median_all, dim=0)
    preds_aux_all = torch.cat(preds_aux_all, dim=0)
    trues_all = torch.cat(trues_all, dim=0)
    x_last_all = torch.cat(x_last_all, dim=0)

    # =========================================================
    # 7. Metrics
    # =========================================================
    print("===== Spatial-Temporal XML Test Metrics =====")

    print("\n--- Diffusion + MEAN ---")
    print("MAE :", mae(preds_mean_all, trues_all).item())
    print("MSE :", mse(preds_mean_all, trues_all).item())
    print("RMSE:", rmse(preds_mean_all, trues_all).item())

    print("\n--- Diffusion + MEDIAN ---")
    print("MAE :", mae(preds_median_all, trues_all).item())
    print("MSE :", mse(preds_median_all, trues_all).item())
    print("RMSE:", rmse(preds_median_all, trues_all).item())

    print("\n--- Auxiliary Head ---")
    print("MAE :", mae(preds_aux_all, trues_all).item())
    print("MSE :", mse(preds_aux_all, trues_all).item())
    print("RMSE:", rmse(preds_aux_all, trues_all).item())

    # =========================================================
    # 8. DEBUG（关键）
    # =========================================================
    print("\n===== DEBUG CHECK =====")

    print("pred_aux shape:", preds_aux_all.shape)
    print("true shape:", trues_all.shape)

    print("pred_aux min/max:", preds_aux_all.min().item(), preds_aux_all.max().item())
    print("true min/max:", trues_all.min().item(), trues_all.max().item())
    print("x_last min/max:", x_last_all.min().item(), x_last_all.max().item())

    print("\n--- Difference Check ---")
    print("aux vs true :", torch.mean(torch.abs(preds_aux_all - trues_all)).item())
    print("aux vs x_last:", torch.mean(torch.abs(preds_aux_all - x_last_all)).item())
    print("true vs x_last:", torch.mean(torch.abs(trues_all - x_last_all)).item())


if __name__ == "__main__":
    main()