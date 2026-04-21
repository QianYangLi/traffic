import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.dataset import load_dataset, split_dataset, normalize, TrafficDataset
from model.gwn import GWNet
from configs.config import HISTORY_LEN, PRED_LEN, BATCH_SIZE, EPOCHS, LR


def build_flow_correlation_graph(train_data, threshold=0.3):
    """
    train_data: [T, N, N]
    build correlation graph on flattened OD flows -> [529, 529]
    """
    T, N, _ = train_data.shape
    flow_data = train_data.reshape(T, -1)  # [T, 529]

    with np.errstate(divide="ignore", invalid="ignore"):
        A = np.corrcoef(flow_data.T)

    A = np.nan_to_num(A)
    A = (A + 1.0) / 2.0
    A[A < threshold] = 0.0
    return A.astype(np.float32)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    data = load_dataset("data/traffic_martixes")

    train_data, val_data, test_data = split_dataset(data)

    train_data, stat_a, stat_b = normalize(train_data)
    val_data, _, _ = normalize(val_data, stat_a, stat_b)

    train_set = TrafficDataset(train_data, HISTORY_LEN, PRED_LEN)
    val_set = TrafficDataset(val_data, HISTORY_LEN, PRED_LEN)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    num_flows = data.shape[1] * data.shape[2]  # 529

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

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_val_loss = float("inf")
    save_path = "best_model_xml_gwn.pt"

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x = x.unsqueeze(-1).to(device)   # [B, T, 529, 1]
            y = y.to(device)                 # [B, 1, 529]

            pred = model(x)                  # [B, 1, 529, T']
            pred = pred[..., -1]             # [B, 1, 529]

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
                x = x.unsqueeze(-1).to(device)
                y = y.to(device)

                pred = model(x)              # [B, 1, 529, T']
                pred = pred[..., -1]         # [B, 1, 529]

                loss = F.smooth_l1_loss(pred, y)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at epoch {epoch}: {save_path}")

        print(f"Epoch {epoch} | Train Loss {train_loss:.6f} | Val Loss {val_loss:.6f}")

    print("Training finished.")
    print("Best Val Loss:", best_val_loss)


if __name__ == "__main__":
    main()