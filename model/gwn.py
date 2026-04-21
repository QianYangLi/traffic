import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d, Conv1d, Conv2d, ModuleList, Parameter


def nconv(x, A):
    return torch.einsum("ncvl,vw->ncwl", (x, A)).contiguous()


class GraphConvNet(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super().__init__()
        c_in = (order * support_len + 1) * c_in
        self.final_conv = Conv2d(
            c_in, c_out, (1, 1), padding=(0, 0), stride=(1, 1), bias=True
        )
        self.dropout = dropout
        self.order = order

    def forward(self, x, support: list):
        out = [x]
        for a in support:
            x1 = nconv(x, a)
            out.append(x1)
            for _ in range(2, self.order + 1):
                x2 = nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.final_conv(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class GWNet(nn.Module):
    def __init__(
        self,
        device,
        num_nodes,
        dropout=0.3,
        supports=None,
        do_graph_conv=True,
        addaptadj=True,
        aptinit=None,
        in_dim=1,
        out_seq_len=1,
        residual_channels=32,
        dilation_channels=32,
        cat_feat_gc=False,
        skip_channels=256,
        end_channels=512,
        kernel_size=2,
        blocks=4,
        layers=2,
        stride=2,
        apt_size=10,
        verbose=0,
    ):
        super().__init__()

        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.skip_channels = skip_channels

        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.do_graph_conv = do_graph_conv
        self.cat_feat_gc = cat_feat_gc
        self.addaptadj = addaptadj
        self.num_nodes = num_nodes
        self.verbose = verbose

        if self.cat_feat_gc:
            self.start_conv = nn.Conv2d(
                in_channels=1, out_channels=residual_channels, kernel_size=(1, 1)
            )
            self.cat_feature_conv = nn.Conv2d(
                in_channels=in_dim - 1,
                out_channels=residual_channels,
                kernel_size=(1, 1),
            )
        else:
            self.start_conv = nn.Conv2d(
                in_channels=in_dim,
                out_channels=residual_channels,
                kernel_size=(1, 1),
            )

        self.fixed_supports = supports or []
        receptive_field = 1

        self.supports_len = len(self.fixed_supports)
        if do_graph_conv and addaptadj:
            if aptinit is None:
                nodevecs = (
                    torch.randn(num_nodes, apt_size),
                    torch.randn(apt_size, num_nodes),
                )
            else:
                nodevecs = self.svd_init(apt_size, aptinit)
            self.supports_len += 1
            self.nodevec1, self.nodevec2 = [
                Parameter(n.to(device), requires_grad=True) for n in nodevecs
            ]

        depth = list(range(blocks * layers))

        self.residual_convs = ModuleList(
            [Conv1d(dilation_channels, residual_channels, 1) for _ in depth]
        )
        self.skip_convs = ModuleList(
            [Conv1d(dilation_channels, skip_channels, 1) for _ in depth]
        )
        self.bn = ModuleList([BatchNorm2d(residual_channels) for _ in depth])
        self.graph_convs = ModuleList(
            [
                GraphConvNet(
                    dilation_channels,
                    residual_channels,
                    dropout,
                    support_len=self.supports_len,
                )
                for _ in depth
            ]
        )

        self.filter_convs = ModuleList()
        self.gate_convs = ModuleList()
        for _ in range(blocks):
            additional_scope = kernel_size - 1
            D = 1
            for _ in range(layers):
                self.filter_convs.append(
                    Conv2d(
                        residual_channels,
                        dilation_channels,
                        (1, kernel_size),
                        dilation=D,
                    )
                )
                self.gate_convs.append(
                    Conv1d(
                        residual_channels,
                        dilation_channels,
                        kernel_size,
                        dilation=D,
                    )
                )
                D *= stride
                receptive_field += additional_scope
                additional_scope *= stride
        self.receptive_field = receptive_field

        self.end_conv_1 = Conv2d(skip_channels, end_channels, (1, 1), bias=True)
        self.end_conv_2 = Conv2d(end_channels, out_seq_len, (1, 1), bias=True)

    @staticmethod
    def svd_init(apt_size, aptinit):
        m, p, n = torch.svd(aptinit)
        nodevec1 = torch.mm(m[:, :apt_size], torch.diag(p[:apt_size] ** 0.5))
        nodevec2 = torch.mm(torch.diag(p[:apt_size] ** 0.5), n[:, :apt_size].t())
        return nodevec1, nodevec2

    def forward(self, x):
        # x: [B, seq_x, num_nodes, in_dim]
        x = x.transpose(1, 3)  # [B, in_dim, num_nodes, seq_x]

        in_len = x.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(x, (self.receptive_field - in_len, 0, 0, 0))

        if self.cat_feat_gc:
            f1, f2 = x[:, [0]], x[:, 1:]
            x1 = self.start_conv(f1)
            x2 = F.leaky_relu(self.cat_feature_conv(f2))
            x = x1 + x2
        else:
            x = self.start_conv(x)

        skip = 0
        adjacency_matrices = self.fixed_supports

        if self.addaptadj:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            adjacency_matrices = self.fixed_supports + [adp]

        for i in range(self.blocks * self.layers):
            residual = x

            filt = torch.tanh(self.filter_convs[i](residual))

            batch_size = residual.size(0)
            gate_in = torch.permute(residual, (0, 2, 1, 3))
            gate_in = torch.reshape(
                gate_in, shape=(batch_size * self.num_nodes, self.residual_channels, -1)
            )
            gate = torch.sigmoid(self.gate_convs[i](gate_in))
            gate = torch.reshape(
                gate, shape=(batch_size, self.num_nodes, self.dilation_channels, -1)
            )
            gate = torch.permute(gate, (0, 2, 1, 3))

            x = filt * gate

            skip_in = torch.permute(x, (0, 2, 1, 3))
            skip_in = torch.reshape(
                skip_in, shape=(batch_size * self.num_nodes, self.dilation_channels, -1)
            )
            s = self.skip_convs[i](skip_in)
            s = torch.reshape(
                s, shape=(batch_size, self.num_nodes, self.skip_channels, -1)
            )
            s = torch.permute(s, (0, 2, 1, 3))

            try:
                skip = skip[:, :, :, -s.size(3):]
            except Exception:
                skip = 0
            skip = s + skip

            if i == (self.blocks * self.layers - 1):
                break

            if self.do_graph_conv:
                graph_out = self.graph_convs[i](x, adjacency_matrices)
                x = x + graph_out if self.cat_feat_gc else graph_out
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)  # [B, out_seq_len, num_nodes, 1]
        return x.squeeze(dim=-1)  # [B, out_seq_len, num_nodes]