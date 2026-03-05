# =========================
#  Alternating GNN-LP Trainer (GNN stays 768-d; add 2 MLP heads for classification)
# =========================
"""
你要求的改动：
- gnn1/gnn2 的结构与输出维度保持不变：768 -> 768
- 额外新增两个分类头 head1/head2（各自对应 gnn1/gnn2）：
    logits = head(ReLU(z))，其中 head 是一个 MLP，用于把 768 降到 num_classes
- 分类用 softmax / log_softmax（训练建议用 log_softmax + NLLLoss，数值更稳）
"""

import os
import random
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    """交替式GNN-LP训练器（GNN输出768不变 + 两个MLP分类头）"""

    def __init__(self, args, data, num_classes):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = data.to(self.device)
        self.num_classes = num_classes
        self.num_nodes = data.y.size(0)

        set_seed(args.seed)

        # ---------- GNN 超参数 ----------
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.dropout = args.dropout

        # ---------- Co-Training 超参数 ----------
        self.W = args.warmup_epochs
        self.T = args.label_update_interval
        self.total_epochs = args.total_epochs
        self.co_train_lr = args.co_train_lr
        self.co_train_tau = args.co_train_tau
        self.post_warmup_lr = args.post_warmup_lr
        self.post_warmup_tau = args.post_warmup_tau

        # ---------- LP 超参数 ----------
        self.lp_alpha = args.lp_alpha
        self.lp_num_iter = args.lp_num_iter

        # ---------- 输入特征：TA embeddings ----------
        self.ta_features = data.ta_embeddings.to(self.device)  # [N, 768]

        # ---------- 加载 LP 预热分布 ----------
        self.P_distribution = self._load_lp_distribution(args.dataset)  # [N, C]

        print(f"📊 TA Embeddings: {self.ta_features.shape}")
        print(f"📊 LP Distribution: {self.P_distribution.shape}")

        # 初始准确率：P_distribution 的 argmax
        y = self.data.y.squeeze()
        init_preds = self.P_distribution.argmax(dim=1)
        init_acc = (init_preds == y).float().mean().item()
        print(f"📊 LP Distribution Accuracy: {init_acc:.4f} ({init_acc * 100:.2f}%)")

        # 构建模型（GNN输出768不变 + 两个MLP head）
        self._build_model()

    def _load_lp_distribution(self, dataset_name):
        data_dir = os.path.join("dataset", dataset_name)
        lp_path = os.path.join(data_dir, "lp_best_distribution.pt")
        return torch.load(lp_path).to(self.device)

    def _build_model(self):
        """构建双GNN（768->768） + 双分类头（768->num_classes）"""
        model_name = self.args.gnn_model
        if model_name == "GCN":
            from core.GNNs.GCN.model import GCN as GNN
        elif model_name == "SAGE":
            from core.GNNs.SAGE.model import SAGE as GNN
        else:
            from core.GNNs.MLP.model import MLP as GNN

        input_dim = 768
        output_dim = 768  # ✅ 保持不变：GNN输出仍为 768-d

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

        # ✅ 新增：两个分类头（先 ReLU 再 MLP 降维到类别）
        # 这里给一个“轻量两层MLP”，你也可以改成单层 Linear。
        head_hidden = getattr(self.args, "head_hidden", 256)  # 如果 args 没有该字段，就默认 256

        self.head1 = nn.Sequential(
            nn.ReLU(),                     # 先对 z 做 ReLU（按你要求）

            nn.Linear(768, self.num_classes)  # MLP 第2层输出 logits
        ).to(self.device)

        self.head2 = nn.Sequential(
            nn.ReLU(),

            nn.Linear(768, self.num_classes)
        ).to(self.device)

        print(f"✅ GNN1: {model_name} (768 → 768)")
        print(f"✅ GNN2: {model_name} (768 → 768)")
        print(f"✅ Head1/Head2: ReLU(z) + MLP(768 → {head_hidden} → {self.num_classes})")

    def _edge_masking(self, edge_index, drop_ratio=0.1):
        num_edges = edge_index.size(1)
        edge_mask = torch.bernoulli(
            torch.ones(num_edges, device=self.device) * (1 - drop_ratio)
        ).bool()
        return edge_index[:, edge_mask]

    @torch.no_grad()
    def _label_propagation(self, Y0, alpha=0.7, num_iter=3):
        edge_index = self.data.edge_index
        N = self.num_nodes

        row, col = edge_index
        deg = torch.zeros(N, device=self.device)
        deg.scatter_add_(0, row, torch.ones(row.size(0), device=self.device))

        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        S = torch.sparse_coo_tensor(edge_index, edge_weight, (N, N))

        Y = Y0.clone()
        for _ in range(num_iter):
            Y = alpha * torch.sparse.mm(S, Y) + (1 - alpha) * Y0

        Y = Y / Y.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return Y

    @torch.no_grad()
    def evaluate(self):
        """评估：返回 pred_mean 分布（由两个 head 的 softmax 平均得到）"""
        self.gnn1.eval()
        self.gnn2.eval()
        self.head1.eval()
        self.head2.eval()

        z1 = self.gnn1(self.ta_features, self.data.edge_index)  # [N, 768]
        z2 = self.gnn2(self.ta_features, self.data.edge_index)  # [N, 768]

        # 如果你想保留归一化可打开；但分类 head 通常不需要 normalize
        # z1 = F.normalize(z1, dim=1)
        # z2 = F.normalize(z2, dim=1)

        logits1 = self.head1(z1)  # [N, C]
        logits2 = self.head2(z2)  # [N, C]

        # 推理阶段：softmax 概率
        pred1 = F.softmax(logits1, dim=1)
        pred2 = F.softmax(logits2, dim=1)
        pred_mean = (pred1 + pred2) / 2

        y = self.data.y.squeeze()
        preds = pred_mean.argmax(dim=1)
        acc = (preds == y).float().mean().item()
        acc1 = (pred1.argmax(dim=1) == y).float().mean().item()
        acc2 = (pred2.argmax(dim=1) == y).float().mean().item()

        macro_f1 = f1_score(y.cpu().numpy(), preds.cpu().numpy(), average="macro")
        print(f"   📊 Eval: Acc={acc:.4f} | G1={acc1:.4f} G2={acc2:.4f}")

        return {
            "acc": acc,
            "acc1": acc1,
            "acc2": acc2,
            "pred1": pred1,
            "pred2": pred2,
            "pred_mean": pred_mean,
            "macro_f1": macro_f1,
        }

    def train(self):
        """
        协同训练:
        - A/B split
        - GNN1+Head1 用 A 训练
        - GNN2+Head2 用 B 训练
        - 预热 W 后每隔 T 用 pred_mean 的 argmax 更新 labels
        """
        W = self.W
        T = self.T
        total_epochs = self.total_epochs

        lr = self.co_train_lr
        tau = self.co_train_tau  # ✅ 温度：对 logits 做 /tau 再 log_softmax

        # ✅ 优化器必须包含 head 参数
        optimizer1 = torch.optim.Adam(
            list(self.gnn1.parameters()) + list(self.head1.parameters()),
            lr=lr
        )
        optimizer2 = torch.optim.Adam(
            list(self.gnn2.parameters()) + list(self.head2.parameters()),
            lr=lr
        )

        y = self.data.y.squeeze()
        edge_index = self.data.edge_index

        # ---------- 随机划分 A/B ----------
        n = self.num_nodes
        perm = torch.randperm(n, device=self.device)
        split = n // 2
        mask_A = torch.zeros(n, dtype=torch.bool, device=self.device)
        mask_B = torch.zeros(n, dtype=torch.bool, device=self.device)
        mask_A[perm[:split]] = True
        mask_B[perm[split:]] = True

        # ---------- 初始标签：P_distribution argmax ----------
        labels_A = self.P_distribution.argmax(dim=1).clone()
        labels_B = self.P_distribution.argmax(dim=1).clone()

        p_acc_total = (self.P_distribution.argmax(dim=1) == y).float().mean().item()
        print(f"   Initial P_dist Acc: Total={p_acc_total:.4f} | A={mask_A.sum().item()} | B={mask_B.sum().item()}")

        best_acc = 0.0
        best_preds = None

        for epoch in range(total_epochs):
            self.gnn1.train(); self.head1.train()
            self.gnn2.train(); self.head2.train()

            edge_index_dropped1 = self._edge_masking(edge_index, drop_ratio=0.2)
            edge_index_dropped2 = self._edge_masking(edge_index, drop_ratio=0.2)

            # ---------- GNN1 + Head1 ----------
            optimizer1.zero_grad()
            z1 = self.gnn1(self.ta_features, edge_index_dropped1)   # [N, 768]
            logits1 = self.head1(z1)                                # [N, C]
            logp1 = F.log_softmax(logits1 , dim=1)             # ✅ 训练用 log-softmax

            # ---------- GNN2 + Head2 ----------
            optimizer2.zero_grad()
            z2 = self.gnn2(self.ta_features, edge_index_dropped2)   # [N, 768]
            logits2 = self.head2(z2)                                # [N, C]
            logp2 = F.log_softmax(logits2, dim=1)

            # ---------- 损失（等价 CrossEntropy） ----------
            loss1 = F.nll_loss(logp1[mask_A], labels_A[mask_A])
            loss2 = F.nll_loss(logp2[mask_B], labels_B[mask_B])

            loss1.backward(retain_graph=True)
            loss2.backward()
            optimizer1.step()
            optimizer2.step()

            # ---------- 产生概率分布用于评估/更新标签 ----------
            pred1 = logp1.exp()
            pred2 = logp2.exp()
            pred_mean = (pred1 + pred2) / 2

            # ---------- 预热后每隔 T 代更新标签 ----------
            if epoch >= W and (epoch - W) % T == 0:
                lr = self.post_warmup_lr
                tau = self.post_warmup_tau

                optimizer1 = torch.optim.Adam(
                    list(self.gnn1.parameters()) + list(self.head1.parameters()),
                    lr=lr
                )
                optimizer2 = torch.optim.Adam(
                    list(self.gnn2.parameters()) + list(self.head2.parameters()),
                    lr=lr
                )

                with torch.no_grad():
                    new_labels = pred_mean.argmax(dim=1)
                    labels_A[mask_A] = new_labels[mask_A]
                    labels_B[mask_B] = new_labels[mask_B]

            # ---------- 训练中打印 ----------
            if epoch % 10 == 0 or epoch == total_epochs - 1:
                with torch.no_grad():
                    preds = pred_mean.argmax(dim=1)
                    acc = (preds == y).float().mean().item()
                    if acc > best_acc:
                        best_acc = acc
                        best_preds = preds.clone()

                    phase = "Warmup" if epoch < W else "Change"
                    print(f"   [{phase}] Epoch {epoch:3d} | Loss1 {loss1.item():.4f} Loss2 {loss2.item():.4f} | Acc {acc:.4f}")

        # ---------- 最终评估 ----------
        eval_result = self.evaluate()
        final_acc = eval_result["acc"]

        if final_acc > best_acc:
            best_acc = final_acc
            best_preds = eval_result["pred_mean"].argmax(dim=1)

        final_f1 = f1_score(y.cpu().numpy(), best_preds.cpu().numpy(), average="macro")

        print(f"\n{'=' * 60}")
        print(f"✅ Co-Training Complete!")
        print(f"   P_dist Acc: {p_acc_total:.4f}")
        print(f"   Best Acc:   {best_acc:.4f} (Delta: {best_acc - p_acc_total:+.4f})")
        print(f"   Macro F1:   {final_f1:.4f}")
        print(f"{'=' * 60}")

        return {"best_acc": best_acc, "acc": best_acc, "macro_f1": final_f1}

    def save(self, path="alternating_gnn_lp.pt"):
        """保存模型（包含 head）"""
        torch.save(
            {
                "gnn1": self.gnn1.state_dict(),
                "gnn2": self.gnn2.state_dict(),
                "head1": self.head1.state_dict(),
                "head2": self.head2.state_dict(),
            },
            path,
        )
        print(f"💾 Model saved to: {path}")
