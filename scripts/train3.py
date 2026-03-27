import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
from torch.utils.data import DataLoader

from data.dataset import (
    load_dataset,
    load_xgz_dataset,
    split_dataset,
    normalize,
    TrafficDataset,
    TrafficVectorDataset
)
from model.predictor import TrafficPredictor
from configs.config import HIDDEN_DIM, HISTORY_LEN, BATCH_SIZE, LR, EPOCHS, DIFFUSION_STEPS


def main():
    # =========================================================
    # 1. Basic settings
    # =========================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # =========================================================
    # 2. Dataset switch
    # =========================================================
    # "xml" -> G?ANT / XML dataset
    # "xgz" -> Abilene / X01~X24.gz dataset
    dataset_type = "xml"

    # =========================================================
    # 3. Multi-step setting
    # =========================================================
    # We predict 3 future time steps at once
    PRED_LEN_3 = 3

    # =========================================================
    # 4. Optional settings for Abilene
    # =========================================================
    # If you want to switch to Abilene, keep feature_type="realOD"
    feature_type = "realOD"

    # If your current normalize() is Min-Max to [0,1], you can keep clamp off in train.
    # No need to clamp here.
    #
    # If in the future you want log1p for Abilene, add it after loading data.
    use_log1p_for_xgz = False

    # =========================================================
    # 5. Load data
    # =========================================================
    if dataset_type == "xml":
        # G?ANT / XML dataset
        data = load_dataset("data/traffic_martixes")
        save_path = "best_model_xml_3step.pt"

    elif dataset_type == "xgz":
        # Abilene / X*.gz dataset
        data = load_xgz_dataset("data/abilene_xgz", feature_type=feature_type)

        if use_log1p_for_xgz:
            import numpy as np
            data = np.log1p(data)

        save_path = "best_model_xgz_3step.pt"

    else:
        raise ValueError(f"Unsupported dataset_type: {dataset_type}")

    # =========================================================
    # 6. Split dataset
    # =========================================================
    train_data, val_data, test_data = split_dataset(data)

    # =========================================================
    # 7. Normalize
    # =========================================================
    # Uses training-set statistics only
    train_data, stat_a, stat_b = normalize(train_data)
    val_data, _, _ = normalize(val_data, stat_a, stat_b)
    test_data, _, _ = normalize(test_data, stat_a, stat_b)

    # =========================================================
    # 8. Build dataset objects
    # =========================================================
    if dataset_type == "xml":
        train_set = TrafficDataset(train_data, HISTORY_LEN, PRED_LEN_3)
        val_set = TrafficDataset(val_data, HISTORY_LEN, PRED_LEN_3)

        # XML data shape: [T, N, N]
        dim = data.shape[1] * data.shape[2]

    else:
        train_set = TrafficVectorDataset(train_data, HISTORY_LEN, PRED_LEN_3)
        val_set = TrafficVectorDataset(val_data, HISTORY_LEN, PRED_LEN_3)

        # XGZ data shape: [T, D]
        dim = data.shape[1]

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    # =========================================================
    # 9. Build model
    # =========================================================
    # Important:
    #   original one-step out_dim = dim
    #   now 3-step out_dim = dim * 3
    model = TrafficPredictor(
        in_dim=dim,
        hidden_dim=HIDDEN_DIM,
        out_dim=dim * PRED_LEN_3
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # =========================================================
    # 10. Train loop
    # =========================================================
    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        # ----------------------
        # Train
        # ----------------------
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)  # [B, hist_len, dim]
            y = y.to(device)  # [B, 3, dim]

            # flatten future 3 steps to [B, 3*dim]
            y = y.reshape(y.size(0), -1)

            t = torch.randint(
                0,
                DIFFUSION_STEPS,
                (x.size(0),),
                device=device
            )

            loss = model(x, y, t)

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

                y = y.reshape(y.size(0), -1)

                t = torch.randint(
                    0,
                    DIFFUSION_STEPS,
                    (x.size(0),),
                    device=device
                )

                loss = model(x, y, t)
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