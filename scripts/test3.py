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
from configs.config import HIDDEN_DIM, HISTORY_LEN, BATCH_SIZE, SAMPLES


def mae(pred, true):
    return torch.mean(torch.abs(pred - true))


def mse(pred, true):
    return torch.mean((pred - true) ** 2)


def confidence_from_relative_uncertainty(step_std, step_mean, eps=1e-6):
    """
    Confidence based on relative uncertainty.

    step_std : [B]
    step_mean: [B]

    confidence = 1 / (1 + std / (abs(mean) + eps))

    Interpretation:
    - smaller relative uncertainty -> higher confidence
    - larger relative uncertainty  -> lower confidence
    """
    rel_uncertainty = step_std / (step_mean.abs() + eps)
    confidence = 1.0 / (1.0 + rel_uncertainty)
    return confidence


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
    PRED_LEN_3 = 3

    # =========================================================
    # 4. Abilene options
    # =========================================================
    feature_type = "realOD"
    use_log1p_for_xgz = False

    # 如果你当前 normalize() 是 Min-Max 到 [0,1]，
    # 并且希望测试输出也限制在 [0,1]，可设为 True
    clamp_output = False

    # =========================================================
    # 5. Load data
    # =========================================================
    if dataset_type == "xml":
        data = load_dataset("data/traffic_martixes")
        model_path = "best_model_xml_3step.pt"

    elif dataset_type == "xgz":
        data = load_xgz_dataset("data/abilene_xgz", feature_type=feature_type)

        if use_log1p_for_xgz:
            import numpy as np
            data = np.log1p(data)

        model_path = "best_model_xgz_3step.pt"

    else:
        raise ValueError(f"Unsupported dataset_type: {dataset_type}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Cannot find {model_path}. Please run train3.py first."
        )

    # =========================================================
    # 6. Split + normalize
    # =========================================================
    train_data, val_data, test_data = split_dataset(data)

    train_data, stat_a, stat_b = normalize(train_data)
    test_data, _, _ = normalize(test_data, stat_a, stat_b)

    # 方便检查归一化范围
    print("Train min/max:", train_data.min(), train_data.max())
    print("Test  min/max:", test_data.min(), test_data.max())

    # =========================================================
    # 7. Build dataset objects
    # =========================================================
    if dataset_type == "xml":
        test_set = TrafficDataset(test_data, HISTORY_LEN, PRED_LEN_3)
        dim = data.shape[1] * data.shape[2]
    else:
        test_set = TrafficVectorDataset(test_data, HISTORY_LEN, PRED_LEN_3)
        dim = data.shape[1]

    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    # =========================================================
    # 8. Build model
    # =========================================================
    model = TrafficPredictor(
        in_dim=dim,
        hidden_dim=HIDDEN_DIM,
        out_dim=dim * PRED_LEN_3
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # =========================================================
    # 9. Containers for 3 future steps
    # =========================================================
    preds_step = [[] for _ in range(PRED_LEN_3)]
    trues_step = [[] for _ in range(PRED_LEN_3)]
    conf_step = [[] for _ in range(PRED_LEN_3)]

    # =========================================================
    # 10. Test loop
    # =========================================================
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)      # [B, hist_len, dim]
            y = y.to(device)      # [B, 3, dim]

            # 多次采样
            samples = model.predict(x, samples=SAMPLES)      # list of [B, 3*dim]
            samples = torch.stack(samples, dim=0)            # [S, B, 3*dim]

            # reshape to [S, B, 3, dim]
            samples = samples.view(SAMPLES, x.size(0), PRED_LEN_3, dim)

            # 用均值作为最终预测
            pred_mean = samples.mean(dim=0)                  # [B, 3, dim]

            if clamp_output:
                pred_mean = torch.clamp(pred_mean, 0.0, 1.0)

            # 多次采样的标准差作为不确定性基础
            pred_std = samples.std(dim=0)                    # [B, 3, dim]

            for k in range(PRED_LEN_3):
                step_pred = pred_mean[:, k, :]               # [B, dim]
                step_true = y[:, k, :]                       # [B, dim]
                step_std = pred_std[:, k, :]                 # [B, dim]

                preds_step[k].append(step_pred.cpu())
                trues_step[k].append(step_true.cpu())

                # -----------------------------
                # 关键修改：相对不确定性置信度
                # -----------------------------
                # 先对每个样本，把当前时刻所有维度的 std / mean 聚合成一个标量
                sample_std = step_std.mean(dim=1)            # [B]
                sample_mean = step_pred.abs().mean(dim=1)    # [B]

                sample_conf = confidence_from_relative_uncertainty(
                    sample_std,
                    sample_mean
                )                                            # [B]

                conf_step[k].append(sample_conf.cpu())

    # =========================================================
    # 11. Merge and evaluate
    # =========================================================
    for k in range(PRED_LEN_3):
        preds_step[k] = torch.cat(preds_step[k], dim=0)
        trues_step[k] = torch.cat(trues_step[k], dim=0)
        conf_step[k] = torch.cat(conf_step[k], dim=0)

    print("===== 3-Step Forecast Metrics =====")
    print(f"Dataset type: {dataset_type}")

    for k in range(PRED_LEN_3):
        step_mae = mae(preds_step[k], trues_step[k]).item()
        step_mse = mse(preds_step[k], trues_step[k]).item()
        step_conf = conf_step[k].mean().item()

        print(f"Step {k+1}:")
        print(f"  MAE              : {step_mae:.6f}")
        print(f"  MSE              : {step_mse:.6f}")
        print(f"  Confidence Score : {step_conf:.6f}")


if __name__ == "__main__":
    main()