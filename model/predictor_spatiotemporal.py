import torch
import torch.nn as nn
import torch.nn.functional as F

from model.encoder import MultiScaleEncoder
from model.diffusion import DiffusionModel
from model.spatial_encoder import SpatialEncoderXML
from model.fusion import ResidualFusion


class TrafficPredictorSpatioTemporal(nn.Module):
    """
    Temporal branch + Spatial branch + Residual fusion + Diffusion head
    + Auxiliary regression head
    """

    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        num_nodes=23,
        A_static=None,
        aux_weight=0.1
    ):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_nodes = num_nodes
        self.aux_weight = aux_weight

        self.temporal_encoder = MultiScaleEncoder(in_dim, hidden_dim)
        self.spatial_encoder = SpatialEncoderXML(
            num_nodes=num_nodes,
            hidden_dim=hidden_dim
        )
        self.fusion = ResidualFusion(hidden_dim)

        # condition dim = hidden_dim + in_dim
        self.diffusion = DiffusionModel(out_dim, hidden_dim + in_dim)

        # auxiliary direct regression head
        self.aux_head = nn.Sequential(
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

    def forward(self, x, y_diff_target, t, y_true=None):
        """
        x: [B, T, D]
        y_diff_target: [B, D]
        t: [B]
        y_true: [B, D] or None
        """
        cond = self.build_condition(x)

        diff_loss = self.diffusion(y_diff_target, cond, t)

        if y_true is not None:
            aux_pred = self.aux_head(cond)
            aux_loss = F.l1_loss(aux_pred, y_true)
            total_loss = diff_loss + self.aux_weight * aux_loss
            return total_loss, diff_loss.detach(), aux_loss.detach()

        return diff_loss, diff_loss.detach(), torch.tensor(0.0, device=x.device)

    @torch.no_grad()
    def predict(self, x, samples=20):
        """
        Diffusion predictions.
        Return: list of [B, D]
        """
        cond = self.build_condition(x)

        preds = []
        for _ in range(samples):
            pred = self.diffusion.sample(cond, (x.size(0), self.out_dim))
            preds.append(pred)

        return preds

    @torch.no_grad()
    def predict_aux(self, x):
        """
        Auxiliary direct regression prediction.
        Return: [B, D]
        """
        cond = self.build_condition(x)
        pred = self.aux_head(cond)
        return pred