import torch
import torch.nn as nn

from model.encoder import MultiScaleEncoder
from model.diffusion import DiffusionModel


class TrafficPredictor(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()

        self.out_dim = out_dim
        self.encoder = MultiScaleEncoder(in_dim, hidden_dim)
        self.diffusion = DiffusionModel(out_dim, hidden_dim)

    def forward(self, x, y, t):
        feat = self.encoder(x)
        cond = feat.mean(dim=1)
        return self.diffusion(y, cond, t)

    @torch.no_grad()
    def predict(self, x, samples=20):
        feat = self.encoder(x)
        cond = feat.mean(dim=1)

        preds = []
        for _ in range(samples):
            pred = self.diffusion.sample(cond, (x.size(0), self.out_dim))
            preds.append(pred)

        return preds