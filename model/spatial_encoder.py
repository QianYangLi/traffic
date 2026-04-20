import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveGraphConv(nn.Module):
    """
    Lightweight adaptive graph convolution.

    Input:
        x: [B, T, N, C]
    Output:
        out: [B, T, N, H]
    """
    def __init__(self, in_dim, out_dim, num_nodes, emb_dim=16):
        super().__init__()

        self.num_nodes = num_nodes
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.input_proj = nn.Linear(in_dim, out_dim)

        # adaptive adjacency
        self.node_emb1 = nn.Parameter(torch.randn(num_nodes, emb_dim))
        self.node_emb2 = nn.Parameter(torch.randn(emb_dim, num_nodes))

        self.output_proj = nn.Linear(out_dim, out_dim)

    def forward(self, x, A_static=None):
        """
        x: [B, T, N, C]
        A_static: [N, N] or None
        """
        B, T, N, C = x.shape

        # learned adaptive adjacency
        A_adp = torch.softmax(F.relu(self.node_emb1 @ self.node_emb2), dim=-1)  # [N, N]

        if A_static is None:
            A = A_adp
        else:
            A = 0.5 * A_static + 0.5 * A_adp

        x = self.input_proj(x)  # [B, T, N, H]

        # graph propagation
        out = torch.einsum("nm,btmc->btnc", A, x)  # [B, T, N, H]
        out = F.relu(self.output_proj(out))

        return out


class SpatialEncoderXML(nn.Module):
    """
    Spatial encoder for XML/G?ANT OD matrix data.

    Steps:
    1. reshape flat OD vector -> [B, T, N, N]
    2. build node features:
         - source feature: row mean
         - destination feature: column mean
    3. graph convolution
    4. node pooling -> [B, T, H]
    """
    def __init__(self, num_nodes=23, hidden_dim=128):
        super().__init__()

        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim

        # each node feature = [src_mean, dst_mean]
        self.gconv1 = AdaptiveGraphConv(in_dim=2, out_dim=hidden_dim, num_nodes=num_nodes)
        self.gconv2 = AdaptiveGraphConv(in_dim=hidden_dim, out_dim=hidden_dim, num_nodes=num_nodes)

    def forward(self, x_flat, A_static=None):
        """
        x_flat: [B, T, N*N]
        return: [B, T, H]
        """
        B, T, D = x_flat.shape
        N = self.num_nodes

        x_mat = x_flat.view(B, T, N, N)  # [B, T, N, N]

        # source outgoing pattern and destination incoming pattern
        src_feat = x_mat.mean(dim=-1, keepdim=True)  # [B, T, N, 1]
        dst_feat = x_mat.mean(dim=-2, keepdim=True)  # [B, T, 1, N]
        dst_feat = dst_feat.transpose(-1, -2)        # [B, T, N, 1]

        node_feat = torch.cat([src_feat, dst_feat], dim=-1)  # [B, T, N, 2]

        h = self.gconv1(node_feat, A_static=A_static)        # [B, T, N, H]
        h = self.gconv2(h, A_static=A_static)                # [B, T, N, H]

        # pool over nodes
        h = h.mean(dim=2)                                    # [B, T, H]

        return h