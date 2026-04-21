import os
import gzip
import xml.etree.ElementTree as ET
import numpy as np
import torch
from torch.utils.data import Dataset


# =========================================================
# Part 1: XML dataset
# =========================================================

def parse_xml_matrix(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    src_nodes = root.findall(".//src")

    node_ids = set()
    for src in src_nodes:
        node_ids.add(int(src.attrib["id"]))
        for dst in src.findall("dst"):
            node_ids.add(int(dst.attrib["id"]))

    node_ids = sorted(list(node_ids))
    node_map = {n: i for i, n in enumerate(node_ids)}

    n = len(node_ids)
    matrix = np.zeros((n, n), dtype=np.float32)

    for src in src_nodes:
        s = node_map[int(src.attrib["id"])]
        for dst in src.findall("dst"):
            d = node_map[int(dst.attrib["id"])]
            matrix[s, d] = float(dst.text)

    return matrix


def load_dataset(folder):
    """
    Load XML traffic matrix dataset.
    Output shape: [T, N, N]
    """
    files = sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(".xml")
    ])

    print("Total XML files:", len(files))

    matrices = []
    for i, f in enumerate(files):
        if i % 500 == 0:
            print("Loading:", i)
        matrices.append(parse_xml_matrix(f))

    data = np.stack(matrices).astype(np.float32)
    print("Dataset shape:", data.shape)
    return data


# =========================================================
# Part 2: X*.gz dataset (Abilene style)
# =========================================================

def parse_xgz_line(line, feature_type="realOD"):
    """
    Parse one line from X*.gz.

    Each line has 720 values = 144 * 5.
    Every 5 values correspond to one OD pair:
        0: realOD
        1: simpleGravityOD
        2: simpleTomogravityOD
        3: generalGravityOD
        4: generalTomogravityOD

    Return shape: [144]
    """
    values = np.array(line.strip().split(), dtype=np.float32)

    if len(values) != 720:
        raise ValueError(f"Expected 720 values per line, got {len(values)}")

    feature_map = {
        "realOD": 0,
        "simpleGravityOD": 1,
        "simpleTomogravityOD": 2,
        "generalGravityOD": 3,
        "generalTomogravityOD": 4,
    }

    if feature_type not in feature_map:
        raise ValueError(f"Unsupported feature_type: {feature_type}")

    idx = feature_map[feature_type]
    od_vector = values[idx::5]

    if len(od_vector) != 144:
        raise ValueError(f"Expected 144 OD values, got {len(od_vector)}")

    return od_vector


def load_xgz_dataset(folder, feature_type="realOD"):
    """
    Load X01~X24.gz style dataset.
    Output shape: [T, 144]
    """
    files = sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.startswith("X") and f.endswith(".gz")
    ])

    print("Total X*.gz files:", len(files))

    all_vectors = []

    for file_idx, f in enumerate(files):
        print(f"Loading file {file_idx + 1}/{len(files)}: {os.path.basename(f)}")

        with gzip.open(f, "rt") as fin:
            for line in fin:
                vec = parse_xgz_line(line, feature_type=feature_type)
                all_vectors.append(vec)

    data = np.stack(all_vectors).astype(np.float32)
    print("XGZ dataset shape:", data.shape)
    return data


# =========================================================
# Part 3: shared utilities
# =========================================================

def split_dataset(data):
    t = len(data)

    train_end = int(t * 0.7)
    val_end = int(t * 0.8)

    train = data[:train_end]
    val = data[train_end:val_end]
    test = data[val_end:]

    print("Train:", train.shape)
    print("Val:", val.shape)
    print("Test:", test.shape)

    return train, val, test


def normalize(data, min_val=None, max_val=None):
    """
    Min-Max normalization, matching sklearn MinMaxScaler behavior:
    normalize each feature/column independently.

    For XML data [T, N, N]:
        reshape to [T, N*N], do per-column Min-Max, then reshape back.
    For vector data [T, D]:
        do per-column Min-Max directly.

    Note:
        This function does NOT clip values, matching sklearn's default transform behavior.
    """

    original_shape = data.shape

    # XML data: [T, N, N] -> [T, N*N]
    if data.ndim == 3:
        T, N, _ = data.shape
        data_2d = data.reshape(T, -1)   # [T, N*N]

    # Vector data: [T, D]
    elif data.ndim == 2:
        data_2d = data                  # [T, D]

    else:
        raise ValueError(f"Unsupported data shape: {data.shape}")

    # fit on training data
    if min_val is None:
        min_val = data_2d.min(axis=0, keepdims=True)   # [1, D]

    if max_val is None:
        max_val = data_2d.max(axis=0, keepdims=True)   # [1, D]

    denom = max_val - min_val
    denom = np.where(denom < 1e-12, 1.0, denom)

    data_2d = (data_2d - min_val) / denom

    # reshape back
    if len(original_shape) == 3:
        data_out = data_2d.reshape(original_shape)
    else:
        data_out = data_2d

    return data_out.astype(np.float32), min_val, max_val


# =========================================================
# Part 4: dataset for XML matrices
# Input: [T, N, N]
# Output:
#   X: [hist_len, N*N]
#   Y: [pred_len, N*N]
# =========================================================

class TrafficDataset(Dataset):
    def __init__(self, data, hist_len=12, pred_len=1):
        t, n, _ = data.shape
        data = data.reshape(t, -1)

        xs = []
        ys = []

        for i in range(t - hist_len - pred_len):
            xs.append(data[i:i + hist_len])
            ys.append(data[i + hist_len:i + hist_len + pred_len])

        self.X = torch.tensor(np.array(xs), dtype=torch.float32)
        self.Y = torch.tensor(np.array(ys), dtype=torch.float32)

        print("Dataset samples:", len(self.X))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# =========================================================
# Part 5: dataset for X*.gz vectors
# Input: [T, D]
# Output:
#   X: [hist_len, D]
#   Y: [pred_len, D]
# =========================================================

class TrafficVectorDataset(Dataset):
    def __init__(self, data, hist_len=12, pred_len=1):
        t, d = data.shape

        xs = []
        ys = []

        for i in range(t - hist_len - pred_len):
            xs.append(data[i:i + hist_len])
            ys.append(data[i + hist_len:i + hist_len + pred_len])

        self.X = torch.tensor(np.array(xs), dtype=torch.float32)
        self.Y = torch.tensor(np.array(ys), dtype=torch.float32)

        print("Vector dataset samples:", len(self.X))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def build_correlation_graph(data, method="row", threshold=None):
    """
    Build adjacency matrix based on node correlation.

    Args:
        data: [T, N, N]
        method: "row" (outgoing) or "col" (incoming)
        threshold: float or None (sparsify graph)

    Returns:
        A: [N, N]
    """

    T, N, _ = data.shape

    if method == "row":
        node_series = data.sum(axis=2)   # [T, N]

    elif method == "col":
        node_series = data.sum(axis=1)   # [T, N]

    else:
        raise ValueError("method must be 'row' or 'col'")

    node_series = node_series.T  # [N, T]

    # correlation
    A = np.corrcoef(node_series)

    # NaN -> 0
    A = np.nan_to_num(A)

    # [-1,1] -> [0,1]
    A = (A + 1) / 2

    # sparsify
    if threshold is not None:
        A[A < threshold] = 0.0

    return A.astype(np.float32)