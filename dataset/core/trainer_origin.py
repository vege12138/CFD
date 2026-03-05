# =========================
#  GNN Trainer (Gated Fusion)
# =========================
"""
GNN训练器 - 简化版
流程:
1. 门控网络融合TA和E嵌入
2. 单GNN编码融合后的嵌入
3. KL损失训练
"""
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from scipy.optimize import linear_sum_assignment
from torch_geometric.utils import add_self_loops

from core.GNNs.label_encoder import LabelEncoder

LOG_FREQ = 10


def set_seed(seed):
    """固定随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class GatingNetwork(nn.Module):
    """门控融合网络: 学习TA和E的融合权重"""
    
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, z_ta, z_e):
        """
        Args:
            z_ta: [N, D] TA嵌入
            z_e: [N, D] E嵌入
        Returns:
            z_fused: [N, D] 融合后的嵌入
        """
        # 拼接TA和E
        concat = torch.cat([z_ta, z_e], dim=1)  # [N, 2D]
        
        # 计算门控权重 α ∈ [0, 1]
        alpha = self.gate(concat)  # [N, 1]
        
        # 融合: z = α * z_ta + (1-α) * z_e
        z_fused = alpha * z_ta + (1 - alpha) * z_e
        
        return z_fused, alpha


class GNNTrainer:
    """GNN训练器 - 门控融合版"""

    def __init__(self, args, data, num_classes):
        """
        Args:
            args: 参数对象
            data: PyG Data对象
            num_classes: 类别数
        """
        self.args = args
        self.device = args.device
        self.seed = args.seed
        set_seed(self.seed)

        # 数据
        self.data = data.to(self.device)
        self.num_nodes = data.y.shape[0]
        self.num_classes = num_classes

        # 配置
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.dropout = args.dropout
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.tau = args.tau
        self.num_conv_layers = args.num_conv_layers

        # 加载嵌入
        self.ta_embeddings = data.ta_embeddings.to(self.device)
        self.e_embeddings = data.e_embeddings.to(self.device)
        self.label_prototypes = F.normalize(data.label_prototypes.to(self.device), p=2, dim=1)
        self.llm_score_matrix = data.llm_score_matrix.to(self.device)

        print(f"📊 TA Embeddings: {self.ta_embeddings.shape}")
        print(f"📊 E Embeddings: {self.e_embeddings.shape}")
        print(f"📊 Label Prototypes: {self.label_prototypes.shape}")
        print(f"📊 LLM Score Matrix: {self.llm_score_matrix.shape}")

        # 构建模型
        self._build_model()

        # 升级LLM得分矩阵
        self._upgrade_llm_scores()

    def _build_model(self):
        """构建模型: 门控网络 + 单GNN + Label Encoder"""
        model_name = self.args.gnn_model

        if model_name == "GCN":
            from core.GNNs.GCN.model import GCN as GNN
        elif model_name == "SAGE":
            from core.GNNs.SAGE.model import SAGE as GNN
        else:
            from core.GNNs.MLP.model import MLP as GNN

        # 门控融合网络
        self.gating_net = GatingNetwork(
            input_dim=768,
            hidden_dim=128
        ).to(self.device)

        # 单GNN (处理融合后的嵌入)
        self.gnn = GNN(
            in_channels=768,
            hidden_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            use_pred=False
        ).to(self.device)

        # Label Encoder
        self.label_encoder = LabelEncoder(
            label_input_dim=768,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            num_layers=2
        ).to(self.device)

        # 优化器
        params = (
            list(self.gating_net.parameters()) +
            list(self.gnn.parameters()) +
            list(self.label_encoder.parameters())
        )
        self.optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)

        print(f"✅ GNN Model: {self.args.gnn_model} (with Gating Fusion)")

    def _normalize_adjacency(self, edge_index, num_nodes):
        """稀疏对称归一化邻接矩阵"""
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        row, col = edge_index[0], edge_index[1]
        deg = torch.bincount(row, minlength=num_nodes).float().to(self.device)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        adj_norm = torch.sparse_coo_tensor(
            indices=edge_index,
            values=edge_weight,
            size=(num_nodes, num_nodes),
            device=self.device,
        ).coalesce()
        return adj_norm

    def _upgrade_llm_scores(self):
        """图传播升级LLM得分矩阵"""
        print("\n🔄 Upgrading LLM Score Matrix...")

        y = self.data.y.squeeze()
        pseudo_y = self.llm_score_matrix.argmax(dim=1)

        llm_acc = (pseudo_y == y).float().mean().item()
        print(f"   LLM Initial Accuracy: {llm_acc:.4f} ({llm_acc * 100:.2f}%)")

        C = self.num_classes
        L = int(self.num_conv_layers)

        if L == 0:
            S0 = self.llm_score_matrix.clamp_min(0.0)
            row_sum = S0.sum(dim=1, keepdim=True).clamp_min(1e-12)
            S0 = S0 / row_sum
            S0 = F.softmax(S0, dim=1)
            self.upgraded_scores = S0
        else:
            adj_norm = self._normalize_adjacency(self.data.edge_index, self.num_nodes)
            H = self.llm_score_matrix
            for _ in range(L):
                H = torch.sparse.mm(adj_norm, H)

            # 伪标签均值中心
            mu_init = torch.zeros(C, H.size(1), device=self.device, dtype=H.dtype)
            for j in range(C):
                mask = (pseudo_y == j)
                if mask.sum() > 0:
                    mu_init[j] = H[mask].mean(dim=0)

            # KMeans聚类
            X = H.detach().float().cpu().numpy()
            init_centers = mu_init.detach().float().cpu().numpy()
            kmeans = KMeans(n_clusters=C, init=init_centers, n_init=1, max_iter=300, random_state=42)
            kmeans.fit(X)
            mu_kmeans = torch.tensor(kmeans.cluster_centers_, device=self.device, dtype=H.dtype)

            # 余弦相似度
            mu_norm = F.normalize(mu_kmeans, dim=1)
            H_norm = F.normalize(H, dim=1)
            similarity = torch.mm(H_norm, mu_norm.T)
            q = F.softmax(similarity, dim=1)

            # 匈牙利匹配
            cluster_assignments_t = similarity.argmax(dim=1)
            cost = torch.zeros(C, C, device=self.device, dtype=torch.long)
            cost.index_put_((cluster_assignments_t, pseudo_y), torch.ones_like(pseudo_y, dtype=torch.long), accumulate=True)
            row_ind, col_ind = linear_sum_assignment((-cost).detach().cpu().numpy())
            label_to_cluster = {int(l): int(c) for c, l in zip(row_ind, col_ind)}
            new_order = [label_to_cluster[j] for j in range(C)]
            self.upgraded_scores = q[:, new_order]

        upgraded_preds = self.upgraded_scores.argmax(dim=1)
        upgraded_acc = (upgraded_preds == y).float().mean().item()
        print(f"   LLM Upgraded Accuracy: {upgraded_acc:.4f} ({upgraded_acc * 100:.2f}%)")
        print(f"   Improvement: {(upgraded_acc - llm_acc) * 100:+.2f}%")

    def _compute_target_distribution(self, q):
        """计算目标分布 P = Q^2 / sum(Q^2)"""
        weight = q ** 2 / (q.sum(dim=0, keepdim=True) + 1e-8)
        return weight / (weight.sum(dim=1, keepdim=True) + 1e-8)

    def _kl_loss(self, p, q):
        """KL散度损失"""
        return F.kl_div(q.log(), p, reduction='batchmean')

    def _forward(self):
        """前向传播: 门控融合 -> GNN编码"""
        # 1. 门控融合TA和E
        z_fused, alpha = self.gating_net(self.ta_embeddings, self.e_embeddings)
        z_fused = F.normalize(z_fused, dim=1)
        
        # 2. GNN编码
        z = self.gnn(z_fused, self.data.edge_index)
        z = F.normalize(z, dim=1)
        
        return z, alpha

    def _train_epoch(self, epoch):
        """单个epoch训练"""
        self.gating_net.train()
        self.gnn.train()
        self.label_encoder.train()
        self.optimizer.zero_grad()

        # 前向传播
        z, alpha = self._forward()

        # 编码原型
        proto_encoded = self.label_encoder(self.label_prototypes)

        # 计算得分矩阵S
        S = torch.mm(z, proto_encoded.T) / self.tau
        S = F.softmax(S, dim=1)

        # 目标分布P
        L = self.upgraded_scores
        P = self._compute_target_distribution(L.detach())

        # KL损失
        loss = self._kl_loss(P, S)

        loss.backward()
        self.optimizer.step()

        # 计算指标
        preds = S.argmax(dim=1)
        acc = (preds == self.data.y.squeeze()).float().mean().item()
        macro_f1 = f1_score(
            self.data.y.squeeze().cpu().numpy(),
            preds.cpu().numpy(),
            average='macro'
        )

        return {
            'loss': loss.item(),
            'acc': acc,
            'macro_f1': macro_f1,
            'alpha_mean': alpha.mean().item()
        }

    @torch.no_grad()
    def evaluate(self):
        """评估"""
        self.gating_net.eval()
        self.gnn.eval()
        self.label_encoder.eval()

        # 前向传播
        z, alpha = self._forward()

        # 原型编码
        proto_encoded = self.label_encoder(self.label_prototypes)

        # 预测
        logits = torch.mm(z, proto_encoded.T) / self.tau
        preds = logits.argmax(dim=1)

        y = self.data.y.squeeze()
        acc = (preds == y).float().mean().item()
        macro_f1 = f1_score(y.cpu().numpy(), preds.cpu().numpy(), average='macro')

        return {
            'acc': acc,
            'macro_f1': macro_f1,
            'alpha_mean': alpha.mean().item()
        }

    def train(self):
        """完整训练"""
        print(f"\n🚀 Starting GNN Training (Gated Fusion)...")
        print(f"   Epochs: {self.epochs}")
        print(f"   LR: {self.lr}\n")

        for epoch in range(self.epochs):
            metrics = self._train_epoch(epoch)

            if epoch % LOG_FREQ == 0:
                print(
                    f"Epoch {epoch:3d} | "
                    f"Loss {metrics['loss']:.4f} | "
                    f"Acc {metrics['acc']:.4f} | "
                    f"F1 {metrics['macro_f1']:.4f} | "
                    f"α_mean {metrics['alpha_mean']:.3f}"
                )

        final = self.evaluate()
        print(f"\n✅ Training Complete!")
        print(f"   Final Acc: {final['acc']:.4f} ({final['acc'] * 100:.2f}%)")
        print(f"   Final Macro-F1: {final['macro_f1']:.4f} ({final['macro_f1'] * 100:.2f}%)")
        print(f"   Final α_mean: {final['alpha_mean']:.3f}")

        return final
