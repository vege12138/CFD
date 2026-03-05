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


class AlternatingGNNLPTrainer:
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

    def _build_model(self):
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
        self.optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)

        # Edge dropping for GNN2
        self.edge_index_dropped = self._edge_masking(self.data.edge_index, drop_ratio=0.1)

        print(f"✅ GNN1: {model_name} (768 → 768)")
        print(f"✅ GNN2: {model_name} (768 → 768)")

    def _edge_masking(self, edge_index, drop_ratio=0.1):
        """随机遮蔽边"""
        num_edges = edge_index.size(1)
        edge_mask = torch.bernoulli(torch.ones(num_edges, device=self.device) * (1 - drop_ratio)).bool()
        edge_index_masked = edge_index[:, edge_mask]
        return edge_index_masked

    @torch.no_grad()
    def _label_propagation(self, Y0, alpha=0.8, num_iter=20):
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

    def _train_epoch(self, target_distribution):
        """单epoch训练，使用指定的目标分布"""
        self.gnn1.train()
        self.gnn2.train()
        self.optimizer.zero_grad()

        # 双GNN前向传播
        z1 = self.gnn1(self.ta_features, self.data.edge_index)
        z2 = self.gnn2(self.ta_features, self.edge_index_dropped)

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

        return {
            'loss': loss.item(),
            'acc': acc,
            'acc1': acc1,
            'acc2': acc2,
            'pred_mean': pred_mean.detach()
        }

    @torch.no_grad()
    def evaluate(self):
        """评估模型，返回pred_mean分布"""
        self.gnn1.eval()
        self.gnn2.eval()

        z1 = self.gnn1(self.ta_features, self.data.edge_index)
        z2 = self.gnn2(self.ta_features, self.edge_index_dropped)

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        proto_norm = F.normalize(self.label_prototypes, dim=1)
        sim1 = torch.mm(z1, proto_norm.T) / self.tau
        sim2 = torch.mm(z2, proto_norm.T) / self.tau

        pred1 = F.softmax(sim1, dim=1)
        pred2 = F.softmax(sim2, dim=1)
        pred_mean = (pred1 + pred2) / 2

        y = self.data.y.squeeze()
        preds = pred_mean.argmax(dim=1)
        acc = (preds == y).float().mean().item()
        acc1 = (pred1.argmax(dim=1) == y).float().mean().item()
        acc2 = (pred2.argmax(dim=1) == y).float().mean().item()

        return {
            'acc': acc,
            'acc1': acc1,
            'acc2': acc2,
            'pred_mean': pred_mean
        }

    def train(self):
        """交替式训练: GNN训练 ↔ LP传播"""
        print(f"{'=' * 60}")
        print(f"🚀 Phase 1: Initial GNN Training with LP Distribution")
        print(f"{'=' * 60}")

        # ===== Phase 1: 使用初始LP分布训练 =====
        current_target = self.P_distribution
        
        for epoch in range(self.epochs):
            metrics = self._train_epoch(current_target)

            if epoch % LOG_FREQ == 0 or epoch == self.epochs - 1:
                print(
                    f"Epoch {epoch:3d} | "
                    f"Loss {metrics['loss']:.4f} | "
                    f"Acc {metrics['acc']:.4f} (G1: {metrics['acc1']:.4f}, G2: {metrics['acc2']:.4f})"
                )

        # 评估并获取LP传播后的分布
        eval_result = self.evaluate()
        print(f"\n✅ Phase 1 Complete! Acc: {eval_result['acc']:.4f}")

        # LP传播pred_mean
        LP_P = self._label_propagation(eval_result['pred_mean'])
        y = self.data.y.squeeze()
        lp_acc = (LP_P.argmax(dim=1) == y).float().mean().item()
        print(f"   LP传播后准确率: {lp_acc:.4f}")

        # ===== Phase 2: 交替训练 =====
        train_LP = 5  # 交替训练轮数
        inner_epochs = 100  # 每轮内部训练epoch数
        best_acc = eval_result['acc']

        for lp_round in range(train_LP):
            print(f"\n{'=' * 60}")
            print(f"🔄 Alternating Round {lp_round + 1}/{train_LP}")
            print(f"{'=' * 60}")

            # 重置模型参数
            self._build_model()
            print(f"   Model reset!")

            # 使用LP_P作为目标分布训练
            current_target = LP_P

            for epoch in range(inner_epochs):
                metrics = self._train_epoch(current_target)

                if epoch % 20 == 0 or epoch == inner_epochs - 1:
                    print(
                        f"   Epoch {epoch:3d} | "
                        f"Loss {metrics['loss']:.4f} | "
                        f"Acc {metrics['acc']:.4f}"
                    )

            # 评估
            eval_result = self.evaluate()
            print(f"\n   Round {lp_round + 1} GNN Acc: {eval_result['acc']:.4f}")

            if eval_result['acc'] > best_acc:
                best_acc = eval_result['acc']

            # LP传播获取新的LP_P
            LP_P = self._label_propagation(eval_result['pred_mean'])
            lp_acc = (LP_P.argmax(dim=1) == y).float().mean().item()
            print(f"   LP传播后准确率: {lp_acc:.4f}")

        # 最终结果
        print(f"\n{'=' * 60}")
        print(f"✅ Alternating Training Complete!")
        print(f"   Best Acc: {best_acc:.4f}")
        print(f"   Final LP Acc: {lp_acc:.4f}")
        print(f"{'=' * 60}")

        return {'best_acc': best_acc, 'final_lp_acc': lp_acc}

    def save(self, path="alternating_gnn_lp.pt"):
        """保存模型"""
        torch.save({
            'gnn1': self.gnn1.state_dict(),
            'gnn2': self.gnn2.state_dict()
        }, path)
        print(f"💾 Model saved to: {path}")
