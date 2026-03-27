import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionModel(nn.Module):
    def __init__(self, data_dim, cond_dim, T=500):
        super().__init__()

        self.T = T

        self.net = nn.Sequential(
            nn.Linear(data_dim + cond_dim + 1, 256),
            nn.ReLU(),
            nn.Linear(256, data_dim)
        )

        beta = torch.linspace(1e-4, 0.02, T)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)

        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)

    def forward(self, x0, cond, t):
        noise = torch.randn_like(x0)

        alpha_bar_t = self.alpha_bar[t].unsqueeze(1)

        xt = torch.sqrt(alpha_bar_t) * x0 + \
             torch.sqrt(1.0 - alpha_bar_t) * noise

        t_embed = t.float().unsqueeze(1) / self.T

        inp = torch.cat([xt, cond, t_embed], dim=1)

        pred = self.net(inp)

        return F.smooth_l1_loss(pred, noise)

    @torch.no_grad()
    def sample(self, cond, shape):
        device = cond.device
        x = torch.randn(shape, device=device)

        for t in reversed(range(self.T)):
            tt = torch.full((shape[0], 1), t / self.T, device=device)

            inp = torch.cat([x, cond, tt], dim=1)
            eps = self.net(inp)

            alpha_t = self.alpha[t]
            alpha_bar_t = self.alpha_bar[t]

            x = (x - (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t) * eps) / torch.sqrt(alpha_t)

        return x