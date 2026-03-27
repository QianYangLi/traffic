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


def normalize(data, min_val=None, max_val=None, clip=True):
    """
    Min-Max normalization to [0,1]
    If test/val values exceed training range, clip them into [0,1].
    """
    if min_val is None:
        min_val = float(data.min())

    if max_val is None:
        max_val = float(data.max())

    if max_val - min_val < 1e-12:
        data = np.zeros_like(data, dtype=np.float32)
    else:
        data = (data - min_val) / (max_val - min_val + 1e-8)

    if clip:
        data = np.clip(data, 0.0, 1.0)

    return data.astype(np.float32), min_val, max_val


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