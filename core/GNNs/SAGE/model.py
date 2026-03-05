# =========================
#  GraphSAGE Model
# =========================
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class SAGE(torch.nn.Module):
    """GraphSAGE Network"""
    
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, use_pred=False):
        super(SAGE, self).__init__()
        self.use_pred = use_pred
        
        if self.use_pred:
            self.encoder = torch.nn.Embedding(out_channels + 1, hidden_channels)
        
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        # 第一层
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        
        # 中间层
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        
        # 最后一层
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):
        if self.use_pred:
            x = self.encoder(x)
            x = torch.flatten(x, start_dim=1)
        
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return x
