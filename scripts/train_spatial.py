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
from configs.config import *


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
    # 3. Split dataset
    # =========================================================
    train_data, val_data, test_data = split_dataset(data)

    # =========================================================
    # 4. Normalize
    # =========================================================
    train_data, stat_a, stat_b = normalize(train_data)
    val_data, _, _ = normalize(val_data, stat_a, stat_b)
    test_data, _, _ = normalize(test_data, stat_a, stat_b)

    # =========================================================
    # 5. Build sliding-window datasets
    # =========================================================
    train_set = TrafficDataset(train_data, HISTORY_LEN, PRED_LEN)
    val_set = TrafficDataset(val_data, HISTORY_LEN, PRED_LEN)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    # =========================================================
    # 6. Build spatial-temporal model
    # =========================================================
    num_nodes = data.shape[1]         # 23
    dim = data.shape[1] * data.shape[2]   # 23*23=529

    # 这里先用单位阵作为静态图
    # 真实拓扑如果后面有，可以直接替换这里
    A_static = torch.eye(num_nodes)

    model = TrafficPredictorSpatioTemporal(
        in_dim=dim,
        hidden_dim=HIDDEN_DIM,
        out_dim=dim,
        num_nodes=num_nodes,
        A_static=A_static
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")
    save_path = "best_model_xml_spatial.pt"

    # =========================================================
    # 7. Train loop
    # =========================================================
    for epoch in range(EPOCHS):
        # ----------------------
        # Train
        # ----------------------
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)                  # [B, hist_len, 529]
            y = y.to(device)                  # [B, 1, 529]

            t = torch.randint(
                0,
                DIFFUSION_STEPS,
                (x.size(0),),
                device=device
            )

            # 单步预测，所以 squeeze(1) -> [B, 529]
            loss = model(x, y.squeeze(1), t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ----------------------
        # Validation
        # ----------------------
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                t = torch.randint(
                    0,
                    DIFFUSION_STEPS,
                    (x.size(0),),
                    device=device
                )

                loss = model(x, y.squeeze(1), t)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # ----------------------
        # Save best model
        # ----------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at epoch {epoch}: {save_path}")

        print(
            f"Epoch {epoch} | Train Loss {train_loss:.6f} | Val Loss {val_loss:.6f}"
        )

    print("Training finished.")
    print("Best Val Loss:", best_val_loss)


if __name__ == "__main__":
    main()