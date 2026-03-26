import sys
import os
import numpy as np

# 把项目根目录加入 Python 路径，保证能找到 data / model / configs
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
from torch.utils.data import DataLoader

from data.dataset import (
    # ===== 原始 XML 数据集相关 =====
    load_dataset,
    TrafficDataset,

    # ===== 新增 X*.gz 数据集相关 =====
    load_xgz_dataset,
    TrafficVectorDataset,

    # ===== 通用函数 =====
    split_dataset,
    normalize
)

from model.predictor import TrafficPredictor
from configs.config import *


def main():
    # =========================================================
    # 1. 设备设置
    # =========================================================
    # 如果机器有 GPU，就用 GPU；否则用 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # =========================================================
    # 2. 数据集切换开关（最关键）
    # =========================================================
    # 你只需要改这里，就可以在两个数据集之间切换
    #
    # 可选:
    #   dataset_type = "xml"
    #   dataset_type = "xgz"
    #
    # 含义:
    #   xml -> 原来的 G?ANT / XML 流量矩阵数据
    #   xgz -> 新的 Abilene X01~X24.gz 数据
    #
    dataset_type = "xgz"

    # =========================================================
    # 3. 加载数据
    # =========================================================
    if dataset_type == "xml":
        # -----------------------------------------------------
        # 原始 XML 数据集
        # 输出形状:
        #   data.shape = [T, N, N]
        # 例如:
        #   [10576, 23, 23]
        # -----------------------------------------------------
        data = load_dataset("data/traffic_martixes")

    elif dataset_type == "xgz":
        # -----------------------------------------------------
        # 新增的 X01~X24.gz 数据集
        # README 说明:
        #   每个 X*.gz 文件中，每一行是一个时间点
        #   每行 720 个数 = 144 * 5
        #   每 5 个值对应同一个 OD 对的 5 种类型:
        #       0: realOD
        #       1: simpleGravityOD
        #       2: simpleTomogravityOD
        #       3: generalGravityOD
        #       4: generalTomogravityOD
        #
        # 这里我们默认只取 realOD，也就是最真实的 OD 流量
        #
        # 输出形状:
        #   data.shape = [T, 144]
        # 例如:
        #   [48384, 144]  # 如果 X01~X24 都齐全
        # -----------------------------------------------------
        data = load_xgz_dataset(
            "data/abilene_xgz",
            feature_type="realOD"
        )
        data = np.log1p(data)

    else:
        raise ValueError(f"Unsupported dataset_type: {dataset_type}")

    # =========================================================
    # 4. 按时间顺序划分训练 / 验证 / 测试集
    # =========================================================
    # 注意:
    #   时间序列任务不能随机切分
    #   必须按时间顺序:
    #       train -> val -> test
    #
    train_data, val_data, test_data = split_dataset(data)

    # =========================================================
    # 5. 数据标准化
    # =========================================================
    # 只用训练集统计量 (mean, std)
    # 然后用同样的 mean/std 去标准化 val/test
    #
    train_data, mean, std = normalize(train_data)
    val_data, _, _ = normalize(val_data, mean, std)
    test_data, _, _ = normalize(test_data, mean, std)

    # =========================================================
    # 6. 构造滑动窗口数据集
    # =========================================================
    # 两个数据集的主要区别在这里：
    #
    # XML 数据集:
    #   输入原始形状 [T, N, N]
    #   在 TrafficDataset 里会 flatten 为 [T, N*N]
    #
    # XGZ 数据集:
    #   输入原始形状 [T, D]
    #   直接用 TrafficVectorDataset
    #
    if dataset_type == "xml":
        train_set = TrafficDataset(train_data, HISTORY_LEN, PRED_LEN)
        val_set = TrafficDataset(val_data, HISTORY_LEN, PRED_LEN)

        # dim = N * N
        dim = data.shape[1] * data.shape[2]

    elif dataset_type == "xgz":
        train_set = TrafficVectorDataset(train_data, HISTORY_LEN, PRED_LEN)
        val_set = TrafficVectorDataset(val_data, HISTORY_LEN, PRED_LEN)

        # dim = 144
        dim = data.shape[1]

    # =========================================================
    # 7. DataLoader
    # =========================================================
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # =========================================================
    # 8. 构建模型
    # =========================================================
    # 不管是 XML 还是 XGZ，核心模型都不改
    # 区别仅仅在输入维度 dim 不同
    #
    # XML:
    #   dim = 23*23 = 529
    #
    # XGZ:
    #   dim = 144
    #
    model = TrafficPredictor(
        in_dim=dim,
        hidden_dim=HIDDEN_DIM,
        out_dim=dim
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # =========================================================
    # 9. 保存最优模型
    # =========================================================
    best_val_loss = float("inf")

    # 如果你后续想加 early stopping，可以在这里扩展
    # patience = 30
    # counter = 0

    # =========================================================
    # 10. 开始训练
    # =========================================================
    for epoch in range(EPOCHS):

        # -------------------------
        # Train
        # -------------------------
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            # diffusion 训练时随机采样时间步 t
            t = torch.randint(
                0,
                DIFFUSION_STEPS,
                (x.size(0),),
                device=device
            )

            # y shape:
            #   [B, pred_len, dim]
            # 当前 pred_len=1，所以 squeeze(1) -> [B, dim]
            loss = model(x, y.squeeze(1), t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # -------------------------
        # Validation
        # -------------------------
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

        # -------------------------
        # Save best model
        # -------------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            # 为了避免两个数据集互相覆盖，模型文件名区分保存
            if dataset_type == "xml":
                save_path = "best_model_xml.pt"
            else:
                save_path = "best_model_xgz.pt"

            torch.save(model.state_dict(), save_path)
            print(f"Model saved at epoch {epoch}: {save_path}")

        # -------------------------
        # Print log
        # -------------------------
        print(
            f"Epoch {epoch} | Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f}"
        )

    print("Training finished.")
    print("Best Val Loss:", best_val_loss)


if __name__ == "__main__":
    main()