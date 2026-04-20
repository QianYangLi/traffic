import torch
import torch.nn as nn

from model.encoder import MultiScaleEncoder
from model.spatial_encoder import SpatialEncoderXML
from model.fusion import ResidualFusion


class TrafficPredictorSpatioTemporalRegression(nn.Module):
    """
    Spatio-temporal encoder + direct regression head
    """

    def __init__(self, in_dim, hidden_dim, out_dim, num_nodes=23, A_static=None):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_nodes = num_nodes

        self.temporal_encoder = MultiScaleEncoder(in_dim, hidden_dim)
        self.spatial_encoder = SpatialEncoderXML(num_nodes=num_nodes, hidden_dim=hidden_dim)
        self.fusion = ResidualFusion(hidden_dim)

        # cond = [pooled feature, x_last]
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim + in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

        if A_static is not None:
            self.register_buffer("A_static", A_static)
        else:
            self.A_static = None

    def build_condition(self, x):
        """
        x: [B, T, D]
        cond: [B, H + D]
        """
        h_t = self.temporal_encoder(x)                          # [B, T, H]
        h_s = self.spatial_encoder(x, A_static=self.A_static)  # [B, T, H]
        h = self.fusion(h_t, h_s)                              # [B, T, H]

        pooled = h.mean(dim=1)                                 # [B, H]
        x_last = x[:, -1, :]                                   # [B, D]

        cond = torch.cat([pooled, x_last], dim=-1)             # [B, H + D]
        return cond

    def forward(self, x):
        """
        x: [B, T, D]
        return: [B, D]
        """
        cond = self.build_condition(x)
        pred = self.reg_head(cond)
        return pred