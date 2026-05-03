# =========================
#  GNN Trainer
# =========================
"""
GNN训练器
流程:
1. 图卷积传播LLM得分矩阵 (l=2)
2. KMeans聚类 + 匈牙利匹配
3. Softmax得到升级后的LLM得分分布
4. 平方作为目标分布P
5. GNN编码 + KL损失训练
"""
import random
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from scipy.optimize import linear_sum_assignment

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


class GNNTrainer:
    """GNN训练器"""

    def __init__(self, args, data, num_classes):
        """
        Args:
            args: 参数对象 (来自OptInit)
            data: PyG Data对象，包含所有嵌入和得分
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
        self.m_ratio = args.m_ratio
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

        # 构建GNN
        self._build_model()

        # 升级LLM得分矩阵
        self._upgrade_llm_scores()

    def _build_model(self):
        """构建GNN模型"""
        model_name = self.args.gnn_model
        
        if model_name == "GCN":
            from core.GNNs.GCN.model import GCN as GNN
        elif model_name == "SAGE":
            from core.GNNs.SAGE.model import SAGE as GNN
        else:
            from core.GNNs.MLP.model import MLP as GNN

        # GNN for TA
        self.gnn_ta = GNN(
            in_channels=768,
            hidden_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            use_pred=False
        ).to(self.device)

        # GNN for E
        self.gnn_e = GNN(
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
            list(self.gnn_ta.parameters()) +
            list(self.gnn_e.parameters()) +
            list(self.label_encoder.parameters())
        )
        self.optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)

        print(f"✅ GNN Model: {self.args.gnn_model}")

    def _normalize_adjacency(self, edge_index, num_nodes):
        """计算对称归一化邻接矩阵"""
        # 添加自环
        edge_index_self = torch.stack([
            torch.arange(num_nodes), 
            torch.arange(num_nodes)
        ], dim=0).to(self.device)
        edge_index = torch.cat([edge_index.to(self.device), edge_index_self], dim=1)

        # 构建邻接矩阵
        adj = torch.zeros(num_nodes, num_nodes, device=self.device)
        adj[edge_index[0], edge_index[1]] = 1.0

        # 对称归一化
        deg = adj.sum(dim=1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        deg_mat = torch.diag(deg_inv_sqrt)
        
        return torch.mm(deg_mat, torch.mm(adj, deg_mat))

    def _upgrade_llm_scores(self):
        """
        升级LLM得分矩阵:
        1. 图卷积传播 (l=2)
        2. KMeans聚类 + 匈牙利匹配
        3. Softmax得到最终分布
        """
        print("\n🔄 Upgrading LLM Score Matrix...")
        
        num_conv_layers = self.num_conv_layers
        
        # 计算LLM初始准确率
        llm_preds = self.llm_score_matrix.argmax(dim=1)
        llm_acc = (llm_preds == self.data.y.squeeze()).float().mean().item()
        print(f"   LLM Initial Accuracy: {llm_acc:.4f} ({llm_acc*100:.2f}%)")

        # 图卷积传播
        adj_norm = self._normalize_adjacency(self.data.edge_index, self.num_nodes)
        scores = self.llm_score_matrix.clone()
        
        for _ in range(num_conv_layers):
            scores = torch.mm(adj_norm, scores)

        # KMeans聚类
        scores_np = scores.cpu().numpy()
        kmeans = KMeans(n_clusters=self.num_classes, random_state=42, n_init=10)
        cluster_assignments = kmeans.fit_predict(scores_np)

        # 计算相似度
        centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(self.device)
        scores_norm = F.normalize(scores, dim=1)
        centers_norm = F.normalize(centers, dim=1)
        similarity = torch.mm(scores_norm, centers_norm.T)
        
        # Softmax得到分布
        cluster_scores = F.softmax(similarity, dim=1)

        # 匈牙利匹配对齐
        llm_preds_np = llm_preds.cpu().numpy()
        cost_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        for c_id, llm_id in zip(cluster_assignments, llm_preds_np):
            cost_matrix[c_id, llm_id] += 1

        row_ind, col_ind = linear_sum_assignment(-cost_matrix)
        cluster_to_llm = {c: l for c, l in zip(row_ind, col_ind)}
        llm_to_cluster = {v: k for k, v in cluster_to_llm.items()}
        
        # 重排列对齐
        new_order = [llm_to_cluster[i] for i in range(self.num_classes)]
        self.upgraded_scores = cluster_scores[:, new_order]

        # 计算升级后准确率
        upgraded_preds = self.upgraded_scores.argmax(dim=1)
        upgraded_acc = (upgraded_preds == self.data.y.squeeze()).float().mean().item()
        print(f"   LLM Upgraded Accuracy: {upgraded_acc:.4f} ({upgraded_acc*100:.2f}%)")
        print(f"   Improvement: {(upgraded_acc - llm_acc)*100:+.2f}%")

    def _compute_target_distribution(self, q):
        """计算目标分布 P = Q^2 / sum(Q^2)"""
        weight = q ** 2 / (q.sum(dim=0, keepdim=True) + 1e-8)
        return weight / (weight.sum(dim=1, keepdim=True) + 1e-8)

    def _kl_loss(self, p, q):
        """KL散度损失"""
        return F.kl_div(q.log(), p, reduction='batchmean')
    def _js_loss(self, p, q, eps=1e-8):
        """
        Jensen–Shannon Divergence: JS(p||q) = 0.5*KL(p||m) + 0.5*KL(q||m),
        where m = 0.5*(p+q)

        p: [N, C] 目标分布（一般 detach）
        q: [N, C] 模型分布（S，有梯度）
        """
        # 防止 log(0)
        p = p.clamp_min(eps)
        q = q.clamp_min(eps)

        m = 0.5 * (p + q)  # [N, C]
        m = m.clamp_min(eps)

        # KL(p||m) = sum p * (log p - log m)
        kl_pm = (p * (p.log() - m.log())).sum(dim=1).mean()

        # KL(q||m) = sum q * (log q - log m)
        kl_qm = (q * (q.log() - m.log())).sum(dim=1).mean()

        return 0.5 * (kl_pm + kl_qm)

    def _forward(self):
        """前向传播"""
        z_ta = self.gnn_ta(self.ta_embeddings, self.data.edge_index)
        z_ta = F.normalize(z_ta, dim=1)

        z_e = self.gnn_e(self.e_embeddings, self.data.edge_index)
        z_e = F.normalize(z_e, dim=1)

        return z_ta, z_e

    def _train_epoch(self, epoch):
        """单个epoch训练"""
        self.gnn_ta.train()
        self.gnn_e.train()
        self.label_encoder.train()
        self.optimizer.zero_grad()

        # GNN编码
        z_ta, z_e = self._forward()

        # ===== 高分节点筛选1: TA/E融合 =====
        # 低置信度节点用z_ta替换z_e
        L = self.upgraded_scores
        X = F.softmax(self.upgraded_scores, dim=1)

        confidence = L.max(dim=1).values
        
        m_ratio = self.m_ratio  # 前m%高置信度节点
        threshold = torch.quantile(confidence, 1 - m_ratio)
        high_conf_mask = confidence >= threshold

        # 高置信度保留z_e，低置信度替换为z_ta
        a = 0.5
        z_e_replaced = torch.where(high_conf_mask.unsqueeze(1), z_e, z_ta)
        z_mean = a * z_ta + (1 - a) * z_e_replaced
        z_mean = F.normalize(z_mean, dim=1)

        # 编码原型
        proto_encoded = self.label_encoder(self.label_prototypes)

        # 计算得分矩阵S
        S = torch.mm(z_mean, proto_encoded.T) / self.tau
        S = F.softmax(S, dim=1)

        # 计算目标分布P (平方升级后的LLM得分)
        #P = self._compute_target_distribution(L.detach())
        P = self._compute_target_distribution(X.detach())

        #P = L.detach()

        # ===== 高分节点筛选2: 目标分布one-hot混合 =====
        # 前5%高置信度节点使用one-hot硬编码
        m_ratio = 0.05  # 前m%高置信度节点
        threshold = torch.quantile(confidence, 1 - m_ratio)
        high_conf_mask = confidence >= threshold

        hard_labels = L.argmax(dim=1)
        P_onehot = F.one_hot(hard_labels, num_classes=self.num_classes).float()

        # 混合: 高置信度用one-hot，其余用软分布
        P = torch.where(high_conf_mask.unsqueeze(1), P_onehot, P)

        # KL损失
        loss = self._kl_loss(P, S)
        #loss = self._js_loss(P, S)


        loss.backward()
        self.optimizer.step()

        # 计算准确率和F1
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
            'macro_f1': macro_f1
        }

    @torch.no_grad()
    def evaluate(self):
        """评估"""
        self.gnn_ta.eval()
        self.gnn_e.eval()
        self.label_encoder.eval()

        z_ta, z_e = self._forward()

        L = self.upgraded_scores
        #L = F.softmax(self.upgraded_scores, dim=1)

        confidence = L.max(dim=1).values

        m_ratio = self.m_ratio  # 前m%高置信度节点
        threshold = torch.quantile(confidence, 1 - m_ratio)
        high_conf_mask = confidence >= threshold

        # 高置信度保留z_e，低置信度替换为z_ta
        a = 0.5
        z_e_replaced = torch.where(high_conf_mask.unsqueeze(1), z_e, z_ta)
        z_mean = a * z_ta + (1 - a) * z_e_replaced
        z_mean = F.normalize(z_mean, dim=1)


        proto_encoded = self.label_encoder(self.label_prototypes)
        logits = torch.mm(z_mean, proto_encoded.T) / self.tau
        preds = logits.argmax(dim=1)

        acc = (preds == self.data.y.squeeze()).float().mean().item()
        macro_f1 = f1_score(
            self.data.y.squeeze().cpu().numpy(),
            preds.cpu().numpy(),
            average='macro'
        )

        return {'acc': acc, 'macro_f1': macro_f1}

    def train(self):
        """完整训练"""
        print(f"\n🚀 Starting GNN Training...")
        print(f"   Epochs: {self.epochs}")
        print(f"   LR: {self.lr}\n")

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
        print(f"   Final Acc: {final['acc']:.4f} ({final['acc']*100:.2f}%)")
        print(f"   Final Macro-F1: {final['macro_f1']:.4f} ({final['macro_f1']*100:.2f}%)")

        return final
