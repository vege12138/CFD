# =========================
#  GAT Model (Official PyG GATConv)
# =========================
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(torch.nn.Module):
    """Multi-layer GAT Network (based on PyG official GATConv)"""

    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 dropout,
                 heads=4,  # 多头数量（中间层）
                 use_pred=False):

        super(GAT, self).__init__()
        self.use_pred = use_pred
        self.dropout = dropout
        self.heads = heads

        # 若输入是离散标签索引，则用Embedding编码
        if self.use_pred:
            self.encoder = torch.nn.Embedding(out_channels + 1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        # =====================
        # 第一层
        # =====================
        # GATConv 输出维度 = hidden_channels * heads (默认 concat=True)
        self.convs.append(
            GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        )
        self.bns.append(
            torch.nn.BatchNorm1d(hidden_channels * heads)
        )

        # =====================
        # 中间层
        # =====================
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels * heads,  # 输入维度 = 上一层输出
                        hidden_channels,
                        heads=heads,
                        dropout=dropout)
            )
            self.bns.append(
                torch.nn.BatchNorm1d(hidden_channels * heads)
            )

        # =====================
        # 最后一层
        # =====================
        # 最后一层通常设置 concat=False
        # 输出维度 = out_channels（而不是 out_channels * heads）
        self.convs.append(
            GATConv(hidden_channels * heads,
                    out_channels,
                    heads=1,
                    concat=False,
                    dropout=dropout)
        )

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):

        # 若输入是标签索引，则转为embedding向量
        if self.use_pred:
            x = self.encoder(x)
            x = torch.flatten(x, start_dim=1)

        # 除最后一层外
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)  # GAT消息传递（带注意力权重）
            x = self.bns[i](x)  # BN
            x = F.elu(x)  # GAT原论文推荐ELU
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 最后一层（不加BN与激活）
        x = self.convs[-1](x, edge_index)

        return x