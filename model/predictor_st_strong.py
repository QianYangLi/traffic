import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_adj(A: torch.Tensor) -> torch.Tensor:
    """
    Row-normalize adjacency.
    A: [N, N]
    """
    rowsum = A.sum(dim=-1, keepdim=True) + 1e-8
    return A / rowsum


class GraphPropagation(nn.Module):
    """
    Multi-support graph propagation:
      static graph + adaptive graph
    """
    def __init__(self, hidden_dim: int, num_nodes: int, dropout: float = 0.0, emb_dim: int = 16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.dropout = dropout

        self.node_emb1 = nn.Parameter(torch.randn(num_nodes, emb_dim))
        self.node_emb2 = nn.Parameter(torch.randn(emb_dim, num_nodes))

        self.mix_proj = nn.Linear(hidden_dim * 3, hidden_dim)

    def forward(self, x: torch.Tensor, A_static: torch.Tensor = None) -> torch.Tensor:
        """
        x: [B, H, N, T]
        return: [B, H, N, T]
        """
        supports = [x]

        if A_static is not None:
            A_s = normalize_adj(A_static)
            x_static = torch.einsum("b h n t, n m -> b h m t", x, A_s)
            supports.append(x_static)

        A_adp = F.softmax(F.relu(self.node_emb1 @ self.node_emb2), dim=-1)
        A_adp = normalize_adj(A_adp)
        x_adp = torch.einsum("b h n t, n m -> b h m t", x, A_adp)
        supports.append(x_adp)

        h = torch.cat(supports, dim=1)              # [B, 3H or 2H, N, T]
        h = h.permute(0, 2, 3, 1)                   # [B, N, T, C]
        h = self.mix_proj(h)                        # [B, N, T, H]
        h = h.permute(0, 3, 1, 2).contiguous()      # [B, H, N, T]
        h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class GatedTemporalBlock(nn.Module):
    """
    Gated dilated temporal conv + graph propagation + residual + skip
    """
    def __init__(self, hidden_dim: int, num_nodes: int, dilation: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.dilation = dilation
        self.dropout = dropout

        self.filter_conv = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size=(1, 2), dilation=(1, dilation)
        )
        self.gate_conv = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size=(1, 2), dilation=(1, dilation)
        )

        self.graph = GraphPropagation(hidden_dim, num_nodes, dropout=dropout)

        self.residual_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 1))
        self.skip_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 1))
        self.bn = nn.BatchNorm2d(hidden_dim)

    def forward(self, x: torch.Tensor, A_static: torch.Tensor = None):
        """
        x: [B, H, N, T]
        return:
          out:  [B, H, N, T']
          skip: [B, H, N, T']
        """
        residual = x

        filt = torch.tanh(self.filter_conv(x))
        gate = torch.sigmoid(self.gate_conv(x))
        h = filt * gate                              # [B, H, N, T']

        h = self.graph(h, A_static=A_static)
        skip = self.skip_proj(h)

        residual_cut = residual[..., -h.size(-1):]
        out = self.residual_proj(h) + residual_cut
        out = self.bn(out)

        return out, skip


class TrafficPredictorSTStrong(nn.Module):
    """
    Strong discriminative spatio-temporal predictor
    - gated dilated temporal conv
    - static correlation graph + adaptive graph
    - skip aggregation
    - residual prediction: y_hat = x_last + delta_hat
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_nodes: int,
        history_len: int,
        A_static: torch.Tensor = None,
        num_blocks: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_nodes = num_nodes
        self.history_len = history_len

        if A_static is not None:
            self.register_buffer("A_static", A_static)
        else:
            self.A_static = None

        # Input is [B, T, D], D = N*N
        # reshape as node-level features: [B, T, N, N]
        # each node feature = concat(outgoing row, incoming col) => 2N
        self.node_feat_dim = num_nodes * 2

        self.input_proj = nn.Conv2d(self.node_feat_dim, hidden_dim, kernel_size=(1, 1))

        dilations = [1, 2, 4, 8, 1, 2][:num_blocks]
        self.blocks = nn.ModuleList([
            GatedTemporalBlock(hidden_dim, num_nodes, d, dropout=dropout) for d in dilations
        ])

        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + in_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, out_dim)
        )

    def build_node_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D] with D=N*N
        return: [B, 2N, N, T]
        """
        B, T, D = x.shape
        N = self.num_nodes
        x_mat = x.view(B, T, N, N)                      # [B, T, N, N]

        row_feat = x_mat                                 # [B, T, N, N]
        col_feat = x_mat.transpose(-1, -2)              # [B, T, N, N]

        node_feat = torch.cat([row_feat, col_feat], dim=-1)   # [B, T, N, 2N]
        node_feat = node_feat.permute(0, 3, 2, 1)             # [B, 2N, N, T]
        return node_feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D]
        return: [B, D]
        """
        B, T, D = x.shape
        x_last = x[:, -1, :]                              # [B, D]

        node_feat = self.build_node_features(x)           # [B, 2N, N, T]
        h = self.input_proj(node_feat)                    # [B, H, N, T]

        skip_list = []
        for block in self.blocks:
            h, skip = block(h, A_static=self.A_static)
            skip_list.append(skip)

        # aggregate skips
        min_t = min(s.size(-1) for s in skip_list)
        skip_sum = 0
        for s in skip_list:
            skip_sum = skip_sum + s[..., -min_t:]

        # use last temporal position + mean temporal position
        h_last = skip_sum[..., -1]                        # [B, H, N]
        h_mean = skip_sum.mean(dim=-1)                    # [B, H, N]

        h_last = h_last.mean(dim=2)                       # [B, H]
        h_mean = h_mean.mean(dim=2)                       # [B, H]

        feat = torch.cat([h_last, h_mean, x_last], dim=-1)   # [B, 2H + D]

        delta_hat = self.output_head(feat)                # [B, D]
        y_hat = x_last + delta_hat                        # residual prediction

        return y_hat