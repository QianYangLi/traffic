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
from model.predictor import TrafficPredictor
from utils.metrics import rmse, mae, mape
from configs.config import HIDDEN_DIM, HISTORY_LEN, PRED_LEN, BATCH_SIZE, SAMPLES


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    data = load_dataset("data/traffic_martixes")

    train_data, val_data, test_data = split_dataset(data)

    train_data, mean, std = normalize(train_data)
    test_data, _, _ = normalize(test_data, mean, std)

    test_set = TrafficDataset(test_data, HISTORY_LEN, PRED_LEN)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    dim = data.shape[1] * data.shape[2]

    model = TrafficPredictor(
        in_dim=dim,
        hidden_dim=HIDDEN_DIM,
        out_dim=dim
    ).to(device)

    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    model.eval()

    preds_all = []
    trues_all = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            samples = model.predict(x, samples=SAMPLES)
            pred = torch.stack(samples, dim=0).mean(dim=0)

            preds_all.append(pred.cpu())
            trues_all.append(y.squeeze(1).cpu())

    preds_all = torch.cat(preds_all, dim=0)
    trues_all = torch.cat(trues_all, dim=0)

    print("MAE :", mae(preds_all, trues_all).item())
    print("RMSE:", rmse(preds_all, trues_all).item())
    print("MAPE:", mape(preds_all, trues_all).item())


if __name__ == "__main__":
    main()