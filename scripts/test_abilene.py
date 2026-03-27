import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import (
    load_xgz_dataset,
    split_dataset,
    normalize,
    TrafficVectorDataset
)
from model.predictor import TrafficPredictor
from utils.metrics import mae, rmse, mape
from configs.config import HIDDEN_DIM, HISTORY_LEN, PRED_LEN, BATCH_SIZE, SAMPLES


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    data = load_xgz_dataset("data/abilene_xgz", feature_type="realOD")

    # 如果你训练时加了 log1p，这里必须同样加；如果没加，这里也不要加
    # data = np.log1p(data)

    train_data, val_data, test_data = split_dataset(data)

    train_data, min_val, max_val = normalize(train_data)
    test_data, _, _ = normalize(test_data, min_val, max_val)

    print("Normalized train min/max:", train_data.min(), train_data.max())
    print("Normalized test min/max:", test_data.min(), test_data.max())

    test_set = TrafficVectorDataset(test_data, HISTORY_LEN, PRED_LEN)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    dim = data.shape[1]

    model = TrafficPredictor(
        in_dim=dim,
        hidden_dim=HIDDEN_DIM,
        out_dim=dim
    ).to(device)

    model.load_state_dict(torch.load("best_model_xgz.pt", map_location=device))
    model.eval()

    preds_all = []
    trues_all = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            samples = model.predict(x, samples=SAMPLES)

            # 用中位数比均值更抗异常值
            pred = torch.stack(samples, dim=0).median(dim=0).values

            # 因为我们定义了归一化范围 [0,1]，这里强制裁剪
            pred = torch.clamp(pred, 0.0, 1.0)

            preds_all.append(pred.cpu())
            trues_all.append(y.squeeze(1).cpu())

    preds_all = torch.cat(preds_all, dim=0)
    trues_all = torch.cat(trues_all, dim=0)

    print("Pred min/max:", preds_all.min().item(), preds_all.max().item())
    print("True min/max:", trues_all.min().item(), trues_all.max().item())

    abs_err = torch.abs(preds_all - trues_all)
    sq_err = (preds_all - trues_all) ** 2

    print("Abs error min/max:", abs_err.min().item(), abs_err.max().item())
    print("Mean abs error:", abs_err.mean().item())
    print("Mean sq error:", sq_err.mean().item())
    print("Top 10 abs error:", torch.topk(abs_err.flatten(), 10).values)

    print("===== Abilene Test Metrics (normalized space) =====")
    print("MAE :", mae(preds_all, trues_all).item())
    print("RMSE:", rmse(preds_all, trues_all).item())
    print("MAPE:", mape(preds_all, trues_all).item())


if __name__ == "__main__":
    main()