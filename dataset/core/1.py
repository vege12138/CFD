# =========================
#  Alternating GNN-LP Trainer
# =========================
"""
交替式GNN-LP训练器:
- 第一阶段：使用LP分布训练GNN
- 评估后获取均值分布，进行LP传播得到LP_P
- 第二阶段循环：重置模型，使用LP_P训练，再LP传播...
- GNN训练和LP传播交替进行
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from pathlib import Path

LOG_FREQ = 10


class GNNTrainer:
    """交替式GNN-LP训练器"""

    def __init__(self, args, data, num_classes):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = data.to(self.device)
        self.num_classes = num_classes
        self.num_nodes = data.y.size(0)

        # 超参数
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.dropout = args.dropout
        self.epochs = args.epochs
        self.tau = args.tau

        # 加载TA嵌入和标签原型
        self.ta_features = data.ta_embeddings.to(self.device)
        self.label_prototypes = data.label_prototypes.to(self.device)
        self.old_label_prototypes = data.label_prototypes.to(self.device)

        # 加载LP预热分布
        self.P_distribution = self._load_lp_distribution(args.dataset).to(self.device)

        print(f"📊 TA Embeddings: {self.ta_features.shape}")
        print(f"📊 Label Prototypes: {self.label_prototypes.shape}")
        print(f"📊 LP Distribution: {self.P_distribution.shape}")

        # 打印初始准确率
        y = self.data.y.squeeze()
        init_preds = self.P_distribution.argmax(dim=1)
        init_acc = (init_preds == y).float().mean().item()
        print(f"📊 LP Distribution Accuracy: {init_acc:.4f} ({init_acc * 100:.2f}%)")

        # 构建模型
        self._build_model()

        print(f"\n{'=' * 60}")
        print(f"📊 Alternating GNN-LP Trainer Initialized")
        print(f"   Nodes: {self.num_nodes}, Classes: {self.num_classes}")
        print(f"   Mode: GNN training ↔ LP propagation alternating")
        print(f"{'=' * 60}\n")

    def _load_lp_distribution(self, dataset_name):
        """加载LP预热分布"""
        data_dir = os.path.join("dataset", dataset_name)
        lp_path = os.path.join(data_dir, "lp_best_distribution.pt")
        lp_dist = torch.load(lp_path).to(self.device)
        return lp_dist

    def _build_model(self, lr=None):
        """构建双GNN模型"""
        model_name = self.args.gnn_model
        if model_name == "GCN":
            from core.GNNs.GCN.model import GCN as GNN
        elif model_name == "SAGE":
            from core.GNNs.SAGE.model import SAGE as GNN
        else:
            from core.GNNs.MLP.model import MLP as GNN

        input_dim = 768
        output_dim = 768

        self.gnn1 = GNN(
            in_channels=input_dim,
            hidden_channels=self.hidden_dim,
            out_channels=output_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            use_pred=False
        ).to(self.device)

        self.gnn2 = GNN(
            in_channels=input_dim,
            hidden_channels=self.hidden_dim,
            out_channels=output_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            use_pred=False
        ).to(self.device)

        params = list(self.gnn1.parameters()) + list(self.gnn2.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.lr if lr == None else lr, weight_decay=self.weight_decay)

        # Edge dropping for GNN2

        print(f"✅ GNN1: {model_name} (768 → 768)")
        print(f"✅ GNN2: {model_name} (768 → 768)")

    def _edge_masking(self, edge_index, drop_ratio=0.1):
        """随机遮蔽边"""
        num_edges = edge_index.size(1)
        edge_mask = torch.bernoulli(torch.ones(num_edges, device=self.device) * (1 - drop_ratio)).bool()
        edge_index_masked = edge_index[:, edge_mask]
        return edge_index_masked

    @torch.no_grad()
    def _label_propagation(self, Y0, alpha=0.9, num_iter=3):
        """标签传播算法"""
        edge_index = self.data.edge_index
        N = self.num_nodes

        row, col = edge_index
        deg = torch.zeros(N, device=self.device)
        deg.scatter_add_(0, row, torch.ones(row.size(0), device=self.device))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        S = torch.sparse_coo_tensor(edge_index, edge_weight, (N, N))

        Y = Y0.clone()
        for t in range(num_iter):
            Y_new = alpha * torch.sparse.mm(S, Y) + (1 - alpha) * Y0
            Y = Y_new

        Y = Y / Y.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return Y

    def _warm_up_train_epoch(self, target_distribution):
        """单epoch训练，使用指定的目标分布"""
        self.gnn1.train()
        self.gnn2.train()
        self.optimizer.zero_grad()

        # 双GNN前向传播
        self.edge_index_dropped1 = self._edge_masking(self.data.edge_index, drop_ratio=0.2)
        self.edge_index_dropped2 = self._edge_masking(self.data.edge_index, drop_ratio=0.2)

        z1 = self.gnn1(self.ta_features, self.edge_index_dropped1)
        z2 = self.gnn2(self.ta_features, self.edge_index_dropped2)

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        proto_norm = F.normalize(self.label_prototypes, dim=1)
        sim1 = torch.mm(z1, proto_norm.T) / self.tau
        sim2 = torch.mm(z2, proto_norm.T) / self.tau

        pred1 = F.softmax(sim1, dim=1)
        pred2 = F.softmax(sim2, dim=1)
        pred_mean = (pred1 + pred2) / 2

        # 使用目标分布的argmax作为标签
        target_y = target_distribution.argmax(dim=1).long()
        tem = (sim1 + sim2) / 2
        loss = F.cross_entropy(tem, target_y)

        loss.backward()
        self.optimizer.step()

        # 评估
        with torch.no_grad():
            preds = pred_mean.argmax(dim=1)
            y = self.data.y.squeeze()
            acc = (preds == y).float().mean().item()
            acc1 = (pred1.argmax(dim=1) == y).float().mean().item()
            acc2 = (pred2.argmax(dim=1) == y).float().mean().item()

            # 三分布标签一致时的准确率
            pred1_labels = pred1.argmax(dim=1)
            pred2_labels = pred2.argmax(dim=1)
            all_agree = (pred1_labels == pred2_labels) & (pred2_labels == target_y)
            agree_count = all_agree.sum().item()
            agree_acc = (y[all_agree] == pred1_labels[all_agree]).float().mean().item() if agree_count > 0 else 0.0

        return {
            'loss': loss.item(),
            'acc': acc,
            'acc1': acc1,
            'acc2': acc2,
            'agree_count': agree_count,
            'agree_acc': agree_acc,
            'pred_mean': pred_mean.detach()
        }

    def js_divergence(self, p, q, eps=1e-12):
        """
        JS(p,q) = 0.5*KL(p||m) + 0.5*KL(q||m), m=0.5*(p+q)
        p,q: [N,C] 概率分布（每行和为1）
        返回: 标量
        """
        p = p.clamp_min(eps)
        q = q.clamp_min(eps)
        m = 0.5 * (p + q)

        kl_pm = (p * (p.log() - m.log())).sum(dim=1)  # [N]
        kl_qm = (q * (q.log() - m.log())).sum(dim=1)  # [N]
        js = 0.5 * (kl_pm + kl_qm)  # [N]
        return js.mean()

    def _edge_smooth_loss(self, z, edge_index, node_weight=None, eps=1e-12):
        """
        边平滑正则：mean_{(i,j)} w_ij * (1 - cos(z_i, z_j))
        z: [N,D]（建议已 normalize）
        edge_index: [2,E]
        node_weight: [N] 可选（如置信度），用于构造边权 w_ij = w_i * w_j
        """
        row, col = edge_index[0], edge_index[1]  # [E], [E]
        cos_ij = (z[row] * z[col]).sum(dim=1).clamp(-1 + eps, 1 - eps)  # [E]
        loss_e = 1.0 - cos_ij  # [E]

        if node_weight is not None:
            w = (node_weight[row] * node_weight[col]).detach()  # [E]
            # 防止全0导致 NaN
            loss = (w * loss_e).sum() / (w.sum() + eps)
        else:
            loss = loss_e.mean()

        return loss

    def _feature_masking(self, X, drop_ratio=0.2):
        """
        随机遮蔽特征维度（Feature Masking）

        目标：实现你给的公式
            1) 随机采样一个向量 M ∈ {0,1}^d
            2) 复制 N 次得到 MF ∈ {0,1}^{N×d}
            3) X_tilde = X ⊙ MF

        参数:
            X: [N, d] 节点特征矩阵（如 ta_features）
            drop_ratio: 遮蔽比例（被置0的特征维度比例）

        返回:
            X_tilde: [N, d] 遮蔽后的特征
            M: [d] 采样的特征维度mask（可用于调试/复现）
        """
        N, d = X.size()

        # 1) 采样 M ∈ {0,1}^d：每个维度以 (1-drop_ratio) 的概率保留
        M = torch.bernoulli(
            torch.full((d,), 1.0 - drop_ratio, device=X.device)
        ).to(dtype=X.dtype)  # [d]，0/1

        # 2) 复制 N 次生成 MF ∈ {0,1}^{N×d}
        MF = M.unsqueeze(0).expand(N, d)  # [N, d]，不额外占显存（view）

        # 3) X̃ = X ⊙ MF
        X_tilde = X * MF

        return X_tilde

    @torch.no_grad()
    def _compute_neighbor_support(self, pred_mean, k=10):
        """
        计算邻居支持度：构建kNN图+原图融合（交集权重为2）
        返回每个节点的邻居支持度得分
        """
        N = self.num_nodes

        # 1. 获取GNN嵌入
        self.gnn1.eval()
        self.gnn2.eval()
        z1 = self.gnn1(self.ta_features, self.data.edge_index)
        z2 = self.gnn2(self.ta_features, self.data.edge_index)
        z_mean = F.normalize((z1 + z2) / 2, dim=1)

        # 2. 构建kNN图
        sim_matrix = torch.mm(z_mean, z_mean.T)
        sim_matrix.fill_diagonal_(-float('inf'))
        _, knn_indices = sim_matrix.topk(k, dim=1)

        knn_adj = torch.zeros(N, N, device=self.device)
        for i in range(N):
            knn_adj[i, knn_indices[i]] = 1
        knn_adj = (knn_adj + knn_adj.T).clamp(max=1)  # 对称化

        # 3. 原始邻接矩阵
        orig_adj = torch.zeros(N, N, device=self.device)
        edge_index = self.data.edge_index
        orig_adj[edge_index[0], edge_index[1]] = 1

        # 4. 融合：交集权重为2，非交集权重为1
        intersection = orig_adj * knn_adj  # 交集
        union = (orig_adj + knn_adj).clamp(max=1)  # 并集
        fused_adj = union + intersection  # 交集部分权重为2

        # 5. 计算邻居支持度
        pred_labels = pred_mean.argmax(dim=1)
        neighbor_support = torch.zeros(N, device=self.device)

        for i in range(N):
            neighbors = fused_adj[i].nonzero(as_tuple=True)[0]
            if len(neighbors) > 0:
                weights = fused_adj[i, neighbors]
                same_label_mask = (pred_labels[neighbors] == pred_labels[i])
                support = (weights * same_label_mask.float()).sum() / weights.sum()
                neighbor_support[i] = support

        return neighbor_support

    def _train_epoch(self, target_distribution, new_clean_mask=None, old_clean_mask=None):
        """单epoch训练，使用指定的目标分布
        new_clean_mask: 新干净节点掩码（权重大）
        old_clean_mask: 旧干净节点掩码（权重小）
        """
        self.gnn1.train()
        self.gnn2.train()
        self.optimizer.zero_grad()

        # 双GNN前向传播
        self.edge_index_dropped1 = self._edge_masking(self.data.edge_index, drop_ratio=0.2)
        self.edge_index_dropped2 = self._edge_masking(self.data.edge_index, drop_ratio=0.2)

        z1 = self.gnn1(self.ta_features, self.edge_index_dropped1)
        z2 = self.gnn2(self.ta_features, self.edge_index_dropped2)

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        proto_norm = F.normalize(self.label_prototypes, dim=1)
        sim1 = torch.mm(z1, proto_norm.T) / self.tau
        sim2 = torch.mm(z2, proto_norm.T) / self.tau

        pred1 = F.softmax(sim1, dim=1)
        pred2 = F.softmax(sim2, dim=1)
        pred_mean = (pred1 + pred2) / 2

        # 分类损失
        target_y = target_distribution.argmax(dim=1).long()  # [N]
        tem = (sim1 + sim2) / 2  # logits [N,C]

        if new_clean_mask is not None or old_clean_mask is not None:
            # 加权损失：新干净节点权重大，旧干净节点权重小
            loss_cls = 0.0
            w = 0.2
            new_weight, old_weight = 1-w, w

            if new_clean_mask is not None and new_clean_mask.sum() > 0:
                loss_new = F.cross_entropy(tem[new_clean_mask], target_y[new_clean_mask])
                loss_cls = loss_cls + new_weight * loss_new

            if old_clean_mask is not None and old_clean_mask.sum() > 0:
                loss_old = F.cross_entropy(tem[old_clean_mask], target_y[old_clean_mask])
                loss_cls = loss_cls + old_weight * loss_old

            # 如果都为空，使用全部节点
            if (new_clean_mask is None or new_clean_mask.sum() == 0) and \
                    (old_clean_mask is None or old_clean_mask.sum() == 0):
                loss_cls = F.cross_entropy(tem, target_y)
        else:
            loss_cls = F.cross_entropy(tem, target_y)

        # 边平滑损失
        conf = target_distribution.max(dim=1).values
        loss_edge_1 = self._edge_smooth_loss(z1, self.edge_index_dropped1, node_weight=conf)
        loss_edge_2 = self._edge_smooth_loss(z2, self.edge_index_dropped2, node_weight=conf)
        loss_edge = 0.5 * (loss_edge_1 + loss_edge_2)

        lambda_edge = 0.1
        loss = loss_cls + lambda_edge * loss_edge


        loss.backward()
        self.optimizer.step()

        # 评估
        with torch.no_grad():
            preds = pred_mean.argmax(dim=1)
            y = self.data.y.squeeze()
            acc = (preds == y).float().mean().item()
            acc1 = (pred1.argmax(dim=1) == y).float().mean().item()
            acc2 = (pred2.argmax(dim=1) == y).float().mean().item()

            # 三分布标签一致时的准确率
            pred1_labels = pred1.argmax(dim=1)
            pred2_labels = pred2.argmax(dim=1)
            all_agree = (pred1_labels == pred2_labels) & (pred2_labels == target_y)
            agree_count = all_agree.sum().item()
            agree_acc = (y[all_agree] == pred1_labels[all_agree]).float().mean().item() if agree_count > 0 else 0.0

        return {
            'loss': loss.item(),
            'acc': acc,
            'acc1': acc1,
            'acc2': acc2,
            'agree_count': agree_count,
            'agree_acc': agree_acc,
            'pred_mean': pred_mean.detach()
        }

    @torch.no_grad()
    def evaluate(self):
        """评估模型，返回pred_mean分布"""
        self.gnn1.eval()
        self.gnn2.eval()

        z1 = self.gnn1(self.ta_features, self.data.edge_index)
        z2 = self.gnn2(self.ta_features, self.data.edge_index)

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        proto_norm = F.normalize(self.label_prototypes, dim=1)
        sim1 = torch.mm(z1, proto_norm.T) / self.tau
        sim2 = torch.mm(z2, proto_norm.T) / self.tau

        # sim1 = torch.mm(z1, proto_norm.T) / self.tau
        # sim2 = torch.mm(z2, proto_norm.T) / self.tau

        pred1 = F.softmax(sim1, dim=1)
        pred2 = F.softmax(sim2, dim=1)
        pred_mean = (pred1 + pred2) / 2

        y = self.data.y.squeeze()
        preds = pred_mean.argmax(dim=1)
        acc = (preds == y).float().mean().item()
        acc1 = (pred1.argmax(dim=1) == y).float().mean().item()
        acc2 = (pred2.argmax(dim=1) == y).float().mean().item()

        # 三分布一致性分析 (GNN1, GNN2, P分布)
        pred1_labels = pred1.argmax(dim=1)
        pred2_labels = pred2.argmax(dim=1)
        p_labels = self.P_distribution.argmax(dim=1)

        three_agree = (pred1_labels == pred2_labels) & (pred2_labels == p_labels)
        three_agree_count = three_agree.sum().item()
        three_agree_acc = (
                y[three_agree] == pred1_labels[three_agree]).float().mean().item() if three_agree_count > 0 else 0.0

        # 四分布一致性分析 (GNN1, GNN2, P, LP_P)
        lp_p = self._label_propagation(pred_mean)
        lp_p_labels = lp_p.argmax(dim=1)

        four_agree = three_agree & (p_labels == lp_p_labels)
        four_agree_count = four_agree.sum().item()
        four_agree_acc = (
                y[four_agree] == pred1_labels[four_agree]).float().mean().item() if four_agree_count > 0 else 0.0

        print(f"   📊 Eval: Acc={acc:.4f} | G1={acc1:.4f} G2={acc2:.4f}")
        print(f"      3分布一致(G1=G2=P): {three_agree_count} Acc={three_agree_acc:.4f}")
        print(f"      4分布一致(G1=G2=P=LP_P): {four_agree_count} Acc={four_agree_acc:.4f}")

        return {
            'acc': acc,
            'acc1': acc1,
            'acc2': acc2,
            'pred1': pred1,
            'pred2': pred2,
            'pred_mean': pred_mean,
            'four_agree': four_agree
        }

    def train(self):
        """交替式训练: GNN训练 ↔ LP传播"""
        print(f"{'=' * 60}")
        print(f"🚀 Alternating GNN-LP Training")
        print(f"{'=' * 60}")

        # 初始化 LP_P 为原始 LP 分布
        LP_P = self.P_distribution
        y = self.data.y.squeeze()

        train_LP = 4  # 交替训练轮数
        inner_epochs = 100  # 每轮内部训练epoch数
        best_acc = 0.0
        lr = 0.0001

        lp_acc = (LP_P.argmax(dim=1) == y).float().mean().item()
        print(f"Initial LP Acc: {lp_acc:.4f}")

        # 初始化干净节点集合
        old_clean_mask = torch.zeros(self.num_nodes, dtype=torch.bool, device=self.device)
        four_agree_mask = None
        eval_result = {}  # 第一轮为空

        for lp_round in range(train_LP):
            # 重置模型参数
            # self.gnn1.reset_parameters()
            # self.gnn2.reset_parameters()

            print(f"\n{'=' * 60}")
            print(f"🔄 Round {lp_round + 1}/{train_LP}")
            print(f"{'=' * 60}")

            # ===== 从第二轮起：先选新干净节点，合并后更新原型 =====
            new_clean_mask = None
            if lp_round > 0 and four_agree_mask is not None:
                # 计算邻居支持度和置信度
                neighbor_support = self._compute_neighbor_support(eval_result['pred_mean'])
                confidence = eval_result['pred_mean'].max(dim=1).values

                # 从候选节点（四分布一致且非旧干净节点）中选取
                candidate_mask = four_agree_mask & (~old_clean_mask)
                candidate_count = candidate_mask.sum().item()

                if candidate_count > 0:
                    candidate_indices = candidate_mask.nonzero(as_tuple=True)[0]
                    candidate_ns_scores = neighbor_support[candidate_indices]
                    candidate_conf_scores = confidence[candidate_indices]
                    candidate_labels = LP_P.argmax(dim=1)[candidate_indices]
                    candidate_true_labels = y[candidate_indices]

                    # 打印不同比例的准确率对比
                    print(f"   📊 候选节点 {candidate_count} 个，各比例准确率对比:")
                    print(f"      比例  |  邻居支持度  |  置信度")
                    for ratio in [0.05, 0.10, 0.30]:
                        select_count = max(1, int(candidate_count * ratio))

                        _, ns_top_idx = candidate_ns_scores.topk(select_count)
                        ns_acc = (candidate_true_labels[ns_top_idx] == candidate_labels[
                            ns_top_idx]).float().mean().item()

                        _, conf_top_idx = candidate_conf_scores.topk(select_count)
                        conf_acc = (candidate_true_labels[conf_top_idx] == candidate_labels[
                            conf_top_idx]).float().mean().item()

                        print(
                            f"      {int(ratio * 100):3d}%  |  {len(ns_top_idx)}  {ns_acc:.4f}    |  {len(conf_top_idx)} {conf_acc:.4f}")

                    # 实际选取前5%
                    select_ratio = 0.05
                    select_count = max(1, int(candidate_count * select_ratio))
                    _, top_indices = candidate_ns_scores.topk(select_count)
                    new_clean_indices = candidate_indices[top_indices]

                    new_clean_mask = torch.zeros(self.num_nodes, dtype=torch.bool, device=self.device)
                    new_clean_mask[new_clean_indices] = True

                    new_clean_acc = (y[new_clean_mask] == LP_P.argmax(dim=1)[new_clean_mask]).float().mean().item()
                    print(f"   📌 选取 {select_count} 个新干净节点 (准确率: {new_clean_acc:.4f})")

                    # 合并新旧干净节点
                    old_clean_mask = old_clean_mask | new_clean_mask
                    print(f"   📌 合并后干净节点: {old_clean_mask.sum().item()} 个")
                else:
                    print(f"   📌 无候选节点可选")

            # ===== 更新原型 =====
            gamma = 0.5
            eps = 1e-12

            if lp_round > 0 and old_clean_mask is not None and old_clean_mask.sum() > 0:
                # 第二轮起：使用合并后的干净节点
                new_prototypes = torch.zeros_like(self.label_prototypes)
                pred_labels = LP_P.argmax(dim=1)

                for c in range(self.num_classes):
                    class_mask = old_clean_mask & (pred_labels == c)
                    if class_mask.sum() > 0:
                        feats = self.ta_features[class_mask]
                        weights = LP_P[:, c][class_mask]
                        weighted_sum = (feats * weights.unsqueeze(1)).sum(dim=0)
                        weight_sum = weights.sum()
                        new_prototypes[c] = weighted_sum / (weight_sum + eps)
                    else:
                        new_prototypes[c] = self.old_label_prototypes[c]

                self.label_prototypes = (1 - gamma) * self.old_label_prototypes + gamma * new_prototypes
                print(f"   Prototypes updated from {old_clean_mask.sum().item()} clean nodes (gamma={gamma})")
            else:
                # 第一轮：使用 LP_P 中每个类别内置信度 > 类别均值的节点
                new_prototypes = torch.zeros_like(self.label_prototypes)
                pred_labels = LP_P.argmax(dim=1)

                for c in range(self.num_classes):
                    # 预测为该类别的节点
                    class_node_mask = (pred_labels == c)
                    if class_node_mask.sum() > 0:
                        # 该类别节点的置信度
                        class_scores = LP_P[:, c][class_node_mask]
                        mean_score = class_scores.mean()
                        # 类别内置信度 > 类别均值的节点
                        high_conf_in_class = class_scores > mean_score

                        if high_conf_in_class.sum() > 0:
                            # 获取原始索引
                            class_indices = class_node_mask.nonzero(as_tuple=True)[0]
                            high_conf_indices = class_indices[high_conf_in_class]

                            feats = self.ta_features[high_conf_indices]
                            weights = LP_P[:, c][high_conf_indices]
                            weighted_sum = (feats * weights.unsqueeze(1)).sum(dim=0)
                            weight_sum = weights.sum()
                            new_prototypes[c] = weighted_sum / (weight_sum + eps)
                        else:
                            new_prototypes[c] = self.old_label_prototypes[c]
                    else:
                        new_prototypes[c] = self.old_label_prototypes[c]

                self.label_prototypes = (1 - gamma) * self.old_label_prototypes + gamma * new_prototypes
                print(f"   Prototypes updated (gamma={gamma})")

            params = list(self.gnn1.parameters()) + list(self.gnn2.parameters())
            self.optimizer = torch.optim.Adam(params, lr=lr, weight_decay=self.weight_decay)
            print(f"   Model & Optimizer reset!")

            # ===== 训练 =====
            current_target = LP_P

            for epoch in range(inner_epochs):
                metrics = self._train_epoch(current_target, new_clean_mask=new_clean_mask,
                                            old_clean_mask=old_clean_mask)

                if epoch % 20 == 0 or epoch == inner_epochs - 1:
                    print(
                        f"   Epoch {epoch:3d} | "
                        f"Loss {metrics['loss']:.4f} | "
                        f"Acc {metrics['acc']:.4f} | "
                        f"Agree: {metrics['agree_count']:4d} Acc={metrics['agree_acc']:.4f}"
                    )

            # 评估
            eval_result = self.evaluate()
            print(f"\n   Round {lp_round + 1} GNN Acc: {eval_result['acc']:.4f}")

            if eval_result['acc'] > best_acc:
                best_acc = eval_result['acc']

            # 获取四分布一致掩码供下一轮使用
            four_agree_mask = eval_result.get('four_agree', None)

            # LP传播获取新的LP_P
            # LP_P = (self._label_propagation(eval_result['pred_mean']) +LP_P)/2
            LP_P = eval_result['pred_mean']

            lp_acc = (LP_P.argmax(dim=1) == y).float().mean().item()
            print(f"   LP传播后准确率: {lp_acc:.4f}")

        # 最终结果
        print(f"\n{'=' * 60}")
        print(f"✅ Alternating Training Complete!")
        print(f"   Best Acc: {best_acc:.4f}")
        print(f"   Final LP Acc: {lp_acc:.4f}")
        print(f"{'=' * 60}")

        return {'best_acc': best_acc, 'acc': lp_acc, 'macro_f1': -1}

    def save(self, path="alternating_gnn_lp.pt"):
        """保存模型"""
        torch.save({
            'gnn1': self.gnn1.state_dict(),
            'gnn2': self.gnn2.state_dict()
        }, path)
        print(f"💾 Model saved to: {path}")
