import torch
import torch.nn as nn

from model.encoder import MultiScaleEncoder
from model.diffusion import DiffusionModel
from model.spatial_encoder import SpatialEncoderXML
from model.fusion import GatedFusion


class TrafficPredictorSpatioTemporal(nn.Module):
    """
    Temporal branch + Spatial branch + Gated fusion + Diffusion head
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_nodes=23, A_static=None):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_nodes = num_nodes

        self.temporal_encoder = MultiScaleEncoder(in_dim, hidden_dim)
        self.spatial_encoder = SpatialEncoderXML(num_nodes=num_nodes, hidden_dim=hidden_dim)
        self.fusion = GatedFusion(hidden_dim)
        self.diffusion = DiffusionModel(out_dim, hidden_dim)

        if A_static is not None:
            self.register_buffer("A_static", A_static)
        else:
            self.A_static = None

    def forward(self, x, y, t):
        """
        x: [B, T, N*N]
        y: [B, out_dim]
        t: [B]
        """
        # temporal branch
        h_t = self.temporal_encoder(x)  # [B, T, H]

        # spatial branch
        h_s = self.spatial_encoder(x, A_static=self.A_static)  # [B, T, H]

        # fusion
        h = self.fusion(h_t, h_s)  # [B, T, H]

        # condition vector
        cond = h.mean(dim=1)  # [B, H]

        return self.diffusion(y, cond, t)

    @torch.no_grad()
    def predict(self, x, samples=20):
        """
        x: [B, T, N*N]
        return: list of [B, out_dim]
        """
        h_t = self.temporal_encoder(x)
        h_s = self.spatial_encoder(x, A_static=self.A_static)
        h = self.fusion(h_t, h_s)

        cond = h.mean(dim=1)

        preds = []
        for _ in range(samples):
            pred = self.diffusion.sample(cond, (x.size(0), self.out_dim))
            preds.append(pred)

        return preds