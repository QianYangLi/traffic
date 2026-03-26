import sys
import os
import numpy as np
# 将项目根目录加入 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
from torch.utils.data import DataLoader

from data.dataset import (
    load_xgz_dataset,
    split_dataset,
    normalize,
    TrafficVectorDataset
)

from model.predictor import TrafficPredictor
from utils.metrics import rmse, mae, mape
from configs.config import HIDDEN_DIM, HISTORY_LEN, PRED_LEN, BATCH_SIZE, SAMPLES


def main():
    # =========================================================
    # 1. 设备设置
    # =========================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # =========================================================
    # 2. 加载 Abilene X01~X24.gz 数据集
    # =========================================================
    # README 说明：
    # - 每个 X*.gz 文件是一周数据
    # - 每行一个时间点
    # - 每行 720 个值 = 144 * 5
    # - 这里默认只取 realOD
    #
    data = load_xgz_dataset(
        "data/abilene_xgz",
        feature_type="realOD"
    )

    # =========================================================
    # 3. 按时间顺序切分训练 / 验证 / 测试
    # =========================================================
    train_data, val_data, test_data = split_dataset(data)

    # =========================================================
    # 4. 标准化
    # =========================================================
    # 注意：
    # 这里只用训练集的 mean / std
    # 然后同样用于 test 集
    #
    train_data, min_val, max_val = normalize(train_data)
    test_data, _, _ = normalize(test_data, min_val, max_val)

    # =========================================================
    # 5. 构造测试集
    # =========================================================
    test_set = TrafficVectorDataset(test_data, HISTORY_LEN, PRED_LEN)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    # =========================================================
    # 6. 构建模型
    # =========================================================
    # Abilene XGZ 数据集:
    #   data.shape = [T, 144]
    # 所以输入维度 dim = 144
    #
    dim = data.shape[1]

    model = TrafficPredictor(
        in_dim=dim,
        hidden_dim=HIDDEN_DIM,
        out_dim=dim
    ).to(device)

    # =========================================================
    # 7. 加载训练好的 Abilene 模型
    # =========================================================
    model_path = "best_model_xgz.pt"

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Cannot find {model_path}. "
            f"Please run training first: python -m scripts.train"
        )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # =========================================================
    # 8. 开始测试
    # =========================================================
    preds_all = []
    trues_all = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            # 扩散模型多次采样，然后取平均作为最终预测
            samples = model.predict(x, samples=SAMPLES)
            pred = torch.stack(samples, dim=0).mean(dim=0)

            preds_all.append(pred.cpu())
            trues_all.append(y.squeeze(1).cpu())

    preds_all = torch.cat(preds_all, dim=0)
    trues_all = torch.cat(trues_all, dim=0)

    # =========================================================
    # 9. 输出指标
    # =========================================================
    print("===== Abilene Test Metrics (normalized space) =====")
    print("MAE :", mae(preds_all, trues_all).item())
    print("RMSE:", rmse(preds_all, trues_all).item())
    print("MAPE:", mape(preds_all, trues_all).item())


if __name__ == "__main__":
    main()  