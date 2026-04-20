import torch
import torch.nn as nn


class GatedFusion(nn.Module):
    """
    Fuse temporal and spatial features:
        h = z * h_t + (1-z) * h_s
    """
    def __init__(self, dim):
        super().__init__()
        self.w_t = nn.Linear(dim, dim)
        self.w_s = nn.Linear(dim, dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, h_t, h_s):
        z = self.sigmoid(self.w_t(h_t) + self.w_s(h_s))
        h = z * h_t + (1.0 - z) * h_s
        return h