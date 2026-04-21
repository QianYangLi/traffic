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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    data = load_dataset("data/traffic_martixes")

    train_data, val_data, test_data = split_dataset(data)

    train_data, stat_a, stat_b = normalize(train_data)
    val_data, _, _ = normalize(val_data, stat_a, stat_b)
    test_data, _, _ = normalize(test_data, stat_a, stat_b)

    train_set = TrafficDataset(train_data, HISTORY_LEN, PRED_LEN)
    val_set = TrafficDataset(val_data, HISTORY_LEN, PRED_LEN)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    num_nodes = data.shape[1]
    dim = data.shape[1] * data.shape[2]

    from data.dataset import build_correlation_graph

# =====================================
# Build correlation graph
# =====================================
    A_np = build_correlation_graph(train_data, method="row", threshold=0.3)

    print("Graph stats:")
    print("A min/max:", A_np.min(), A_np.max())
    print("Non-zero ratio:", (A_np > 0).mean())

    A_static = torch.tensor(A_np, dtype=torch.float32).to(device)

    model = TrafficPredictorSpatioTemporal(
        in_dim=dim,
        hidden_dim=HIDDEN_DIM,
        out_dim=dim,
        num_nodes=num_nodes,
        A_static=A_static,
        aux_weight=0.1
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")
    save_path = "best_model_xml_spatial.pt"

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_diff_loss = 0.0
        train_aux_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)                  # [B, hist_len, 529]
            y = y.to(device)                  # [B, 1, 529]

            x_last = x[:, -1, :]
            y_true = y.squeeze(1)

            # diffusion Ñ§²Ð²î
            delta_target = y_true - x_last

            t = torch.randint(
                0,
                DIFFUSION_STEPS,
                (x.size(0),),
                device=device
            )

            loss, diff_loss, aux_loss = model(
                x,
                delta_target,
                t,
                y_true=y_true
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_diff_loss += diff_loss.item()
            train_aux_loss += aux_loss.item()

        train_loss /= len(train_loader)
        train_diff_loss /= len(train_loader)
        train_aux_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        val_diff_loss = 0.0
        val_aux_loss = 0.0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                x_last = x[:, -1, :]
                y_true = y.squeeze(1)
                delta_target = y_true - x_last

                t = torch.randint(
                    0,
                    DIFFUSION_STEPS,
                    (x.size(0),),
                    device=device
                )

                loss, diff_loss, aux_loss = model(
                    x,
                    delta_target,
                    t,
                    y_true=y_true
                )

                val_loss += loss.item()
                val_diff_loss += diff_loss.item()
                val_aux_loss += aux_loss.item()

        val_loss /= len(val_loader)
        val_diff_loss /= len(val_loader)
        val_aux_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at epoch {epoch}: {save_path}")

        print(
            f"Epoch {epoch} | "
            f"Train Total {train_loss:.6f} | Train Diff {train_diff_loss:.6f} | Train Aux {train_aux_loss:.6f} | "
            f"Val Total {val_loss:.6f} | Val Diff {val_diff_loss:.6f} | Val Aux {val_aux_loss:.6f}"
        )

    print("Training finished.")
    print("Best Val Loss:", best_val_loss)


if __name__ == "__main__":
    main()