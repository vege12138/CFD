# =========================
#  GNN Trainer (DEC-style Clustering)
# =========================
"""
GNN训练器 - DEC风格聚类

流程:
1. LLM得分 × 原型嵌入 = 加权原型表示 [N, 768]
2. GNN编码 → [N, 256]
3. 使用LLM伪类初始化聚类中心
4. KMeans优化中心
5. Student's t分布计算软分配 Q
6. KL散度损失: L = KL(P || Q), P为LLM归一化得分
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


class GNNTrainer:
    """GNN训练器 - DEC风格聚类"""

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
        self.hidden_dim = args.hidden_dim  # 256
        self.num_layers = args.num_layers
        self.dropout = args.dropout
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.alpha = 1  # Student's t分布自由度参数 v

        # 加载嵌入
        self.ta_embeddings = data.ta_embeddings.to(self.device)  # [N, 768]
        self.label_prototypes = data.label_prototypes.to(self.device)  # [C, 768]
        self.llm_score_matrix = data.llm_score_matrix.to(self.device)  # [N, C]

        print(f"📊 TA Embeddings: {self.ta_embeddings.shape}")
        print(f"📊 Label Prototypes: {self.label_prototypes.shape}")
        print(f"📊 LLM Score Matrix: {self.llm_score_matrix.shape}")

        # 计算融合得分和目标分布P
        self._compute_fused_scores()

        # 计算原型加权输入
        self._compute_prototype_weighted_input()

        # 构建模型
        self._build_model()

        # 初始化聚类中心
        self._init_cluster_centers()

    def _compute_fused_scores(self):
        """计算融合得分: 0.9*LLM + 0.1*TA相似度"""
        y = self.data.y.squeeze()

        # 1. LLM初始准确率
        llm_norm = self.llm_score_matrix.clamp_min(0.0)
        row_sum = llm_norm.sum(dim=1, keepdim=True).clamp_min(1e-12)
        llm_norm = llm_norm / row_sum  # [N, C]

        llm_preds = llm_norm.argmax(dim=1)
        llm_acc = (llm_preds == y).float().mean().item()
        print(f"📊 LLM Initial Accuracy: {llm_acc:.4f} ({llm_acc * 100:.2f}%)")

        # 2. TA嵌入与原型相似度
        ta_norm = F.normalize(self.ta_embeddings, dim=1)  # [N, 768]
        proto_norm = F.normalize(self.label_prototypes, dim=1)  # [C, 768]
        ta_sim = torch.mm(ta_norm, proto_norm.T)  # [N, C]
        ta_scores = F.softmax(ta_sim, dim=1)  # [N, C]

        ta_preds = ta_scores.argmax(dim=1)
        ta_acc = (ta_preds == y).float().mean().item()
        print(f"📊 TA-Prototype Similarity Accuracy: {ta_acc:.4f} ({ta_acc * 100:.2f}%)")

        # 3. 融合: 0.9*LLM + 0.1*TA
        self.target_P = 1 * llm_norm #+ 0.1 * ta_scores  # [N, C]

        fused_preds = self.target_P.argmax(dim=1)
        fused_acc = (fused_preds == y).float().mean().item()
        print(f"📊 Fused Score Accuracy (0.9*LLM + 0.1*TA): {fused_acc:.4f} ({fused_acc * 100:.2f}%)")

    def _compute_prototype_weighted_input(self):
        """计算原型加权输入: 融合得分 × 原型嵌入"""
        # 使用融合后的得分作为注意力权重
        attn_weights = self.target_P  # [N, C]

        # 加权求和: [N, C] × [C, 768] = [N, 768]
        self.x_input = torch.mm(attn_weights, self.label_prototypes)  # [N, 768]
        print(f"📊 Prototype-Weighted Input: {self.x_input.shape}")

    def _build_model(self):
        """构建GNN模型"""
        model_name = self.args.gnn_model

        if model_name == "GCN":
            from core.GNNs.GCN.model import GCN as GNN
        elif model_name == "SAGE":
            from core.GNNs.SAGE.model import SAGE as GNN
        else:
            from core.GNNs.MLP.model import MLP as GNN

        # GNN编码器: 768 → 256
        self.gnn = GNN(
            in_channels=768,
            hidden_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            use_pred=False
        ).to(self.device)

        # 聚类中心 (可学习参数)
        self.cluster_centers = nn.Parameter(
            torch.zeros(self.num_classes, self.hidden_dim, device=self.device)
        )

        # 优化器
        params = list(self.gnn.parameters()) + [self.cluster_centers]
        self.optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)

        print(f"✅ GNN Model: {self.args.gnn_model}")
        print(f"✅ Cluster Centers: {self.cluster_centers.shape}")

    def _init_cluster_centers(self):
        """使用LLM伪类 + KMeans初始化聚类中心"""
        print("\n🔄 Initializing Cluster Centers...")

        # GNN前向传播获取初始嵌入
        with torch.no_grad():
            self.gnn.eval()
            h = self.gnn(self.x_input, self.data.edge_index)  # [N, 256]
            h = F.normalize(h, dim=1)

        # LLM伪标签
        pseudo_y = self.target_P.argmax(dim=1)

        # 按伪类计算均值中心
        mu_init = torch.zeros(self.num_classes, self.hidden_dim, device=self.device)
        for j in range(self.num_classes):
            mask = (pseudo_y == j)
            if mask.sum() > 0:
                mu_init[j] = h[mask].mean(dim=0)
            else:
                random_idx = torch.randint(0, self.num_nodes, (1,), device=self.device)
                mu_init[j] = h[random_idx].squeeze()

        # KMeans优化中心
        print("   Running KMeans...")
        X = h.detach().cpu().numpy()
        init_centers = mu_init.detach().cpu().numpy()

        kmeans = KMeans(
            n_clusters=self.num_classes,
            init=init_centers,
            n_init=1,
            max_iter=300,
            random_state=42
        )
        kmeans.fit(X)

        # 匈牙利匹配对齐
        cluster_assignments = torch.tensor(kmeans.labels_, device=self.device)
        cost = torch.zeros(self.num_classes, self.num_classes, device=self.device, dtype=torch.long)
        cost.index_put_(
            (cluster_assignments, pseudo_y),
            torch.ones_like(pseudo_y, dtype=torch.long),
            accumulate=True
        )
        row_ind, col_ind = linear_sum_assignment((-cost).cpu().numpy())
        cluster_to_label = {int(c): int(l) for c, l in zip(row_ind, col_ind)}

        # 重排中心
        new_centers = torch.zeros_like(mu_init)
        for c, l in cluster_to_label.items():
            new_centers[l] = torch.tensor(kmeans.cluster_centers_[c], device=self.device)

        # 初始化聚类中心参数
        with torch.no_grad():
            self.cluster_centers.copy_(new_centers)

        print(f"   Cluster Centers initialized: {self.cluster_centers.shape}")

        # 计算初始准确率
        self._evaluate_initial()

    def _evaluate_initial(self):
        """评估初始聚类准确率"""
        with torch.no_grad():
            self.gnn.eval()
            h = self.gnn(self.x_input, self.data.edge_index)
            h = F.normalize(h, dim=1)
            q = self._student_t_distribution(h)
            preds = q.argmax(dim=1)
            y = self.data.y.squeeze()
            acc = (preds == y).float().mean().item()
            print(f"   Initial Clustering Accuracy: {acc:.4f} ({acc * 100:.2f}%)")

    def _student_t_distribution(self, h):
        """
        Student's t分布计算软分配

        q_ij = (1 + ||h_i - μ_j||² / α)^(-(α+1)/2) / Σ_j'(...)

        Args:
            h: [N, D] 节点嵌入

        Returns:
            q: [N, C] 软分配概率
        """
        # ||h_i - μ_j||² : [N, C]
        dist_sq = torch.cdist(h, self.cluster_centers, p=2).pow(2)  # [N, C]

        # (1 + ||h_i - μ_j||² / α)^(-(α+1)/2)
        numerator = (1.0 + dist_sq / self.alpha).pow(-(self.alpha + 1) / 2)

        # 归一化
        q = numerator / (numerator.sum(dim=1, keepdim=True) + 1e-12)

        return q

    def _kl_loss(self, p, q):
        """KL散度损失: KL(P || Q)"""
        # 防止log(0)
        q = q.clamp_min(1e-12)
        p = p.clamp_min(1e-12)
        return (p * (p.log() - q.log())).sum(dim=1).mean()

    def _forward(self):
        """前向传播"""
        h = self.gnn(self.x_input, self.data.edge_index)  # [N, 256]
        h = F.normalize(h, dim=1)
        return h

    def _train_epoch(self, epoch):
        """单个epoch训练"""
        self.gnn.train()
        self.optimizer.zero_grad()

        # GNN编码
        h = self._forward()
        # if epoch == 400:
        #     self._init_cluster_centers()
        # Student's t分布软分配
        q = self._student_t_distribution(h)

        # 目标分布P
        p = self.target_P.detach()

        # KL损失
        loss = self._kl_loss(p, q)

        loss.backward()
        self.optimizer.step()

        # 计算准确率
        preds = q.argmax(dim=1)
        y = self.data.y.squeeze()
        acc = (preds == y).float().mean().item()
        macro_f1 = f1_score(y.cpu().numpy(), preds.detach().cpu().numpy(), average='macro')

        return {
            'loss': loss.item(),
            'acc': acc,
            'macro_f1': macro_f1
        }

    @torch.no_grad()
    def evaluate(self):
        """评估"""
        self.gnn.eval()

        h = self._forward()
        q = self._student_t_distribution(h)

        preds = q.argmax(dim=1)
        y = self.data.y.squeeze()

        acc = (preds == y).float().mean().item()
        macro_f1 = f1_score(y.cpu().numpy(), preds.cpu().numpy(), average='macro')

        return {
            'acc': acc,
            'macro_f1': macro_f1
        }

    def train(self):
        """完整训练"""
        print(f"\n🚀 Starting GNN Training (DEC-style)...")
        print(f"   Epochs: {self.epochs}")
        print(f"   LR: {self.lr}")
        print(f"   Alpha (t-dist): {self.alpha}\n")

        for epoch in range(self.epochs):
            metrics = self._train_epoch(epoch)

            if epoch % LOG_FREQ == 0:
                print(
                    f"Epoch {epoch:3d} | "
                    f"Loss {metrics['loss']:.4f} | "
                    f"Acc {metrics['acc']:.4f} | "
                    f"F1 {metrics['macro_f1']:.4f}"
                )

        final = self.evaluate()
        print(f"\n✅ Training Complete!")
        print(f"   Final Acc: {final['acc']:.4f} ({final['acc'] * 100:.2f}%)")
        print(f"   Final Macro-F1: {final['macro_f1']:.4f} ({final['macro_f1'] * 100:.2f}%)")

        return final
