import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn.functional as F
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
from configs.config import HISTORY_LEN, PRED_LEN, BATCH_SIZE, EPOCHS, LR


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
    val_data, _, _ = normalize(val_data, stat_a, stat_b)

    train_set = TrafficDataset(train_data, HISTORY_LEN, PRED_LEN)
    val_set = TrafficDataset(val_data, HISTORY_LEN, PRED_LEN)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    input_dim = data.shape[1] * data.shape[2]   # 529

    model = build_model(model_name, input_dim=input_dim, hidden_dim=128).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")
    save_path = f"best_model_xml_{model_name}.pt"

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)                # [B, T, 529]
            y = y.to(device).squeeze(1)     # [B, 529]

            pred = model(x)                 # [B, 529]
            loss = F.smooth_l1_loss(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device).squeeze(1)

                pred = model(x)
                loss = F.smooth_l1_loss(pred, y)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at epoch {epoch}: {save_path}")

        print(f"[{model_name}] Epoch {epoch} | Train Loss {train_loss:.6f} | Val Loss {val_loss:.6f}")

    print("Training finished.")
    print("Best Val Loss:", best_val_loss)


if __name__ == "__main__":
    main()