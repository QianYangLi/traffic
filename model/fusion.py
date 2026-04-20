import torch
import torch.nn as nn


class ResidualFusion(nn.Module):
    """
    Temporal feature as main branch, spatial feature as enhancement.

    h = h_t + z * proj(h_s)
    """
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

    def forward(self, h_t, h_s):
        z = self.gate(torch.cat([h_t, h_s], dim=-1))
        h = h_t + z * self.proj(h_s)
        return h