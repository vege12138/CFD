# =========================
#  Simple GNN Trainer
# =========================
"""
简单GNN训练器:
- 输入: TA特征 (768维)
- 使用平方后的目标分布P²作为软标签
- 软交叉熵损失训练
"""
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score

LOG_FREQ = 10


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class GNNTrainer:
    """简单GNN训练器"""

    def __init__(self, args, data, num_classes):
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

        # 加载TA特征
        self.ta_features = data.ta_embeddings.to(self.device)

        # 预处理LLM得分矩阵，获取平方后的目标分布
        self.P_distribution = self._preprocess_llm_scores(
            data.llm_score_matrix.to(self.device),
            data.edge_index.to(self.device)
        )

        print(f"📊 TA Embeddings: {self.ta_features.shape}")
        print(f"📊 P Distribution (squared): {self.P_distribution.shape}")

        # 构建模型
        self._build_model()

        print(f"\n{'=' * 60}")
        print(f"📊 Simple GNN Trainer Initialized")
        print(f"   Nodes: {self.num_nodes}, Classes: {self.num_classes}")
        print(f"   Input: TA features (768)")
        print(f"   Loss: Soft Cross-Entropy with P²")
        print(f"{'=' * 60}\n")

    def _compute_sym_norm_adj(self, edge_index, num_nodes):
        """计算对称归一化邻接矩阵"""
        from torch_geometric.utils import add_self_loops

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
            device=self.device
        ).coalesce()

        return adj_norm

    def _compute_target_distribution(self, q):
        """计算目标分布 P = q^2 / sum(q)"""
        weight = q ** 2 / (q.sum(dim=0, keepdim=True) + 1e-8)
        p = weight / (weight.sum(dim=1, keepdim=True) + 1e-8)
        return p

    def _preprocess_llm_scores(self, llm_scores, edge_index):
        """预处理LLM得分矩阵，返回平方后的目标分布"""
        print("\n🔄 Preprocessing LLM Score Matrix...")
        y = self.data.y.squeeze()

        pseudo_y = llm_scores.argmax(dim=1)
        init_acc = (pseudo_y == y).float().mean().item()
        print(f"   Initial LLM Accuracy: {init_acc:.4f} ({init_acc * 100:.2f}%)")

        # 图卷积平滑
        label_prototypes = self.data.label_prototypes.to(self.device)
        P = llm_scores
        H = torch.mm(P, label_prototypes)

        num_conv_layers = 2
        adj_norm = self._compute_sym_norm_adj(edge_index, self.num_nodes)
        for _ in range(num_conv_layers):
            H = torch.sparse.mm(adj_norm, H)

        # 计算聚类中心
        C = self.num_classes
        mu_init = torch.zeros(C, H.size(1), device=self.device, dtype=H.dtype)

        for j in range(C):
            mask = (pseudo_y == j)
            if mask.sum() > 0:
                mu_init[j] = H[mask].mean(dim=0)
            else:
                mu_init[j] = label_prototypes[j]

        with torch.no_grad():
            Hn = F.normalize(H, dim=1)
            mun = F.normalize(mu_init, dim=1)
            sim_conv = torch.mm(Hn, mun.T) / 0.1
            q_conv = F.softmax(sim_conv, dim=1)
            acc_conv = (q_conv.argmax(dim=1) == y).float().mean().item()
            print(f"   [After Conv] Acc: {acc_conv:.4f} ({acc_conv * 100:.2f}%)")

        # 平方后的目标分布
        P_squared = self._compute_target_distribution(q_conv)
        acc_squared = (P_squared.argmax(dim=1) == y).float().mean().item()
        print(f"   [After Square] Acc: {acc_squared:.4f} ({acc_squared * 100:.2f}%)")

        return P_squared

    def _build_model(self):
        """构建GNN模型"""
        model_name = self.args.gnn_model
        if model_name == "GCN":
            from core.GNNs.GCN.model import GCN as GNN
        elif model_name == "SAGE":
            from core.GNNs.SAGE.model import SAGE as GNN
        else:
            from core.GNNs.MLP.model import MLP as GNN

        # 输入: TA特征 (768), 输出: 嵌入 (768)
        input_dim = 768
        embed_dim = 768

        # GNN: TA → 768维嵌入
        self.gnn = GNN(
            in_channels=input_dim,
            hidden_channels=self.hidden_dim,
            out_channels=embed_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            use_pred=False
        ).to(self.device)

        # 加载标签原型
        self.label_prototypes = self.data.label_prototypes.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.gnn.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        print(f"✅ GNN: {model_name} ({input_dim} → {embed_dim})")
        print(f"✅ Label Prototypes: {self.label_prototypes.shape}")

    def _kl_loss(self, p, q):
        """KL散度损失: KL(p||q)"""
        p = p.clamp_min(1e-12)
        q = q.clamp_min(1e-12)
        kl = (p * (p.log() - q.log())).sum(dim=1).mean()
        return kl

    def _soft_ce_loss(self, logits, target_dist):
        """软交叉熵损失: -sum(P * log(softmax(logits)))"""
        log_probs = F.log_softmax(logits, dim=1)
        loss = -(target_dist * log_probs).sum(dim=1).mean()
        return loss

    def _train_epoch(self):
        """单epoch训练"""
        self.gnn.train()
        self.optimizer.zero_grad()

        # GNN前向传播 → 嵌入
        z = self.gnn(self.ta_features, self.data.edge_index)  # [N, 768]
        z = F.normalize(z, dim=1)

        # 计算与原型的余弦相似度 → 概率分布
        proto_norm = F.normalize(self.label_prototypes, dim=1)
        sim = torch.mm(z, proto_norm.T) / self.tau  # [N, C]
        pred_dist = F.softmax(sim, dim=1)

        # KL损失
        #loss = self._kl_loss(self.P_distribution, pred_dist)
        loss = self._soft_ce_loss(self.P_distribution, pred_dist)

        loss.backward()
        self.optimizer.step()

        # 评估
        with torch.no_grad():
            preds = pred_dist.argmax(dim=1)
            y = self.data.y.squeeze()
            acc = (preds == y).float().mean().item()
            macro_f1 = f1_score(y.cpu().numpy(), preds.cpu().numpy(), average='macro')

        return {
            'loss': loss.item(),
            'acc': acc,
            'macro_f1': macro_f1
        }

    @torch.no_grad()
    def evaluate(self):
        """评估模型"""
        self.gnn.eval()

        # GNN前向传播 → 嵌入
        z = self.gnn(self.ta_features, self.data.edge_index)
        z = F.normalize(z, dim=1)

        # 计算与原型的余弦相似度 → 概率分布
        proto_norm = F.normalize(self.label_prototypes, dim=1)
        sim = torch.mm(z, proto_norm.T) / self.tau
        pred_dist = F.softmax(sim, dim=1)
        preds = pred_dist.argmax(dim=1)

        y = self.data.y.squeeze()
        acc = (preds == y).float().mean().item()
        macro_f1 = f1_score(y.cpu().numpy(), preds.cpu().numpy(), average='macro')

        return {
            'acc': acc,
            'macro_f1': macro_f1
        }

    def train(self):
        """训练"""
        print(f"\n🚀 Starting Simple GNN Training...")
        print(f"   Epochs: {self.epochs}")
        print(f"   LR: {self.lr}")
        print(f"   Loss: Soft Cross-Entropy\n")

        print(f"{'=' * 50}")
        print(f"Training: TA → GNN → Soft-CE(P²)")
        print(f"{'=' * 50}")

        for epoch in range(self.epochs):
            metrics = self._train_epoch()
            if epoch % LOG_FREQ == 0:
                print(
                    f"Epoch {epoch:3d} | "
                    f"Loss {metrics['loss']:.4f} | "
                    f"Acc {metrics['acc']:.4f}"
                )

        final = self.evaluate()
        print(f"\n✅ Training Complete!")
        print(f"   Final Acc: {final['acc']:.4f} ({final['acc'] * 100:.2f}%)")
        print(f"   Final Macro-F1: {final['macro_f1']:.4f}")

        return final

    def save(self, path="simple_gnn.pt"):
        """保存模型"""
        torch.save({
            'gnn': self.gnn.state_dict()
        }, path)
        print(f"💾 Model saved to: {path}")
