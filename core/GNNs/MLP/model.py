# =========================
#  MLP Model
# =========================
import torch
import torch.nn.functional as F


class MLP(torch.nn.Module):
    """Multi-Layer Perceptron (无图结构)"""
    
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, use_pred=False):
        super(MLP, self).__init__()
        self.use_pred = use_pred
        
        if self.use_pred:
            self.encoder = torch.nn.Embedding(out_channels + 1, hidden_channels)
        
        self.layers = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        # 第一层
        self.layers.append(torch.nn.Linear(in_channels, hidden_channels))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        
        # 中间层
        for _ in range(num_layers - 2):
            self.layers.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        
        # 最后一层
        self.layers.append(torch.nn.Linear(hidden_channels, out_channels))
        
        self.dropout = dropout

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index=None):
        """edge_index参数保留但不使用，以保持接口一致"""
        if self.use_pred:
            x = self.encoder(x)
            x = torch.flatten(x, start_dim=1)
        
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.layers[-1](x)
        return x
