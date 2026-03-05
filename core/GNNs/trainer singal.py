
# =========================
#  Single-GNN Trainer (Ablation)
# =========================
"""
消融版目标：
1) 只保留一个GNN（无 co-training / 无双分支）
2) 不做 edge masking（使用原始 edge_index）
3) 直接用 self.P_distribution 作为监督信号训练 500 epoch
4) 每隔 10 epoch 做一次验证（优先用 val_mask；没有就用 test_mask；再没有就全图）
5) 训练结束后加载 best_val 模型，仅做一次最终评估（同上优先级）
"""

import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score


LOG_FREQ = 10


def set_seed(seed: int):
    # 固定随机种子，保证消融可复现
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class GNNTrainer:
    """单GNN + 用 P_distribution 监督训练"""

    def __init__(self, args, data, num_classes: int):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = data.to(self.device)
        self.num_classes = num_classes
        self.num_nodes = self.data.y.size(0)

        set_seed(getattr(args, "seed", 0))

        # ========== 超参数（给默认值，避免args缺字段） ==========
        self.hidden_dim = getattr(args, "hidden_dim", 256)
        self.num_layers = getattr(args, "num_layers", 2)
        self.dropout = getattr(args, "dropout", 0.5)

        # 学习率/温度：沿用你原先命名（co_train_lr/co_train_tau），没有就退化到 lr/tau/默认值
        self.lr = getattr(args, "co_train_lr", getattr(args, "lr", 1e-3))
        self.tau = getattr(args, "co_train_tau", getattr(args, "tau", 1.0))

        # 固定训练轮数：按你要求 500
        self.total_epochs = 500

        # ========== 输入特征/原型 ==========
        self.ta_features = self.data.ta_embeddings.to(self.device)          # [N, d]
        self.label_prototypes = self.data.label_prototypes.to(self.device)  # [C, d]
        assert self.label_prototypes.size(0) == self.num_classes

        # ========== 监督分布 P ==========
        dataset_name = getattr(args, "dataset", None)
        self.P_distribution = self._load_lp_distribution(dataset_name)      # [N, C]
        self.P_distribution = self._row_normalize(self.P_distribution)

        # 打印 P 分布初始准确率（用于 sanity check）
        y = self.data.y.squeeze()
        init_preds = self.P_distribution.argmax(dim=1)
        init_acc = (init_preds == y).float().mean().item()
        print(f"📊 P_distribution Accuracy (all nodes): {init_acc:.4f} ({init_acc * 100:.2f}%)")

        # ========== 构建单GNN ==========
        self._build_model()

        # 优化器
        self.optimizer = torch.optim.Adam(self.gnn.parameters(), lr=self.lr)

    # ---------------------------
    # utilities
    # ---------------------------
    def _row_normalize(self, mat: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        # 行归一化：确保每个节点的概率和为1（避免KL数值问题）
        return mat / mat.sum(dim=1, keepdim=True).clamp_min(eps)

    def _get_eval_mask(self):
        """
        评估mask：全节点
        """
        return torch.ones(self.num_nodes, dtype=torch.bool, device=self.device), "all_nodes"

    def _get_train_mask(self):
        """
        训练mask：全节点
        """
        return torch.ones(self.num_nodes, dtype=torch.bool, device=self.device), "all_nodes"

    # ---------------------------
    # load/build
    # ---------------------------
    def _load_lp_distribution(self, dataset_name: str):
        # 读取你原逻辑：dataset/<name>/lp_best_distribution.pt
        if dataset_name is None:
            raise ValueError("args.dataset 不存在，无法定位 lp_best_distribution.pt")

        data_dir = os.path.join("dataset", str(dataset_name))
        lp_path = os.path.join(data_dir, "lp_best_distribution.pt")
        if not os.path.exists(lp_path):
            raise FileNotFoundError(f"找不到：{lp_path}")

        lp_dist = torch.load(lp_path, map_location=self.device)
        # 期望形状 [N, C]
        return lp_dist.to(self.device)

    def _build_model(self):
        # 单GNN：沿用你原选择逻辑
        model_name = getattr(self.args, "gnn_model", "GCN")
        if model_name == "GCN":
            from core.GNNs.GCN.model import GCN as GNN
        elif model_name == "SAGE":
            from core.GNNs.SAGE.model import SAGE as GNN
        else:
            from core.GNNs.MLP.model import MLP as GNN

        in_dim = self.ta_features.size(1)
        out_dim = self.label_prototypes.size(1)  # 必须与原型维度一致（用于相似度分类）

        self.gnn = GNN(
            in_channels=in_dim,
            hidden_channels=self.hidden_dim,
            out_channels=out_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            use_pred=False
        ).to(self.device)

        print(f"✅ Single GNN: {model_name} ({in_dim} → {out_dim}), layers={self.num_layers}, hid={self.hidden_dim}")

    # ---------------------------
    # forward/loss/eval
    # ---------------------------
    def _forward_logits(self):
        """
        前向得到 logits（节点×类别），不做边掩码：直接用原始 edge_index
        - 先GNN得到节点表示 z
        - 与原型做余弦相似度（归一化后点积）得到 logits
        """
        x = self.ta_features
        edge_index = self.data.edge_index

        # 兼容 MLP 可能不需要 edge_index 的情况
        try:
            z = self.gnn(x, edge_index)
        except TypeError:
            z = self.gnn(x)

        z = F.normalize(z, dim=1)
        proto = F.normalize(self.label_prototypes, dim=1)
        logits = (z @ proto.t()) / float(self.tau)
        return logits


    @torch.no_grad()
    def evaluate(self, split_mask=None):
        """
        评估：
        - 输出 Acc / Macro-F1（在指定mask上）
        - 默认用 _get_eval_mask() 规则选验证集合
        """
        self.gnn.eval()
        y = self.data.y.squeeze()

        logits = self._forward_logits()
        pred_prob = F.softmax(logits, dim=1)
        preds = pred_prob.argmax(dim=1)

        if split_mask is None:
            split_mask, split_name = self._get_eval_mask()
        else:
            split_name = "custom_mask"

        y_s = y[split_mask]
        p_s = preds[split_mask]

        acc = (p_s == y_s).float().mean().item()
        macro = f1_score(y_s.detach().cpu().numpy(), p_s.detach().cpu().numpy(), average="macro")

        return {
            "acc": acc,
            "macro_f1": macro,
            "split": split_name,
        }

    # ---------------------------
    # train
    # ---------------------------
    def train(self):
        """
        训练流程（按你要求）：
        - 训练500 epoch
        - 每10 epoch 验证一次
        - 保存 best_val_acc 的模型参数
        - 训练结束加载 best，再做一次最终评估
        """
        train_mask, train_name = self._get_train_mask()
        eval_mask, eval_name = self._get_eval_mask()

        y = self.data.y.squeeze()
        p_acc_total = (self.P_distribution.argmax(dim=1) == y).float().mean().item()
        print(f"{'='*60}")
        print(f"🚀 Ablation Train: Single-GNN, no edge masking, soft P supervision")
        print(f"   epochs={self.total_epochs}, lr={self.lr}, tau={self.tau}")
        print(f"   train={train_name}, eval={eval_name}")
        print(f"   P_dist Acc (all) = {p_acc_total:.4f}")
        print(f"{'='*60}")

        best_val_acc = -1.0
        best_state = None

        for epoch in range(1, self.total_epochs + 1):
            self.gnn.train()
            self.optimizer.zero_grad()

            logits = self._forward_logits()

            # 用 LP 硬标签做交叉熵损失（全节点）
            hard_labels = self.P_distribution.argmax(dim=1)  # [N]
            loss = F.cross_entropy(logits, hard_labels)

            loss.backward()
            self.optimizer.step()

            # 每10代验证
            if epoch % LOG_FREQ == 0 or epoch == 1:
                self.gnn.eval()
                with torch.no_grad():
                    # 验证只看 eval_mask（val/test/all）
                    val_res = self.evaluate(split_mask=eval_mask)

                if val_res["acc"] > best_val_acc:
                    best_val_acc = val_res["acc"]
                    best_state = {k: v.detach().cpu().clone() for k, v in self.gnn.state_dict().items()}

                print(f"[Epoch {epoch:3d}] loss={loss.item():.4f} | "
                      f"{val_res['split']} Acc={val_res['acc']:.4f} MacroF1={val_res['macro_f1']:.4f} | "
                      f"best_acc={best_val_acc:.4f}")

        # 训练结束：加载 best
        if best_state is not None:
            self.gnn.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})

        final_res = self.evaluate(split_mask=eval_mask)
        print(f"{'='*60}")
        print(f"✅ Done. Best {eval_name} Acc={best_val_acc:.4f} | "
              f"Final({final_res['split']}) Acc={final_res['acc']:.4f} MacroF1={final_res['macro_f1']:.4f}")
        print(f"{'='*60}")

        return {
            "best_acc": best_val_acc,
            "acc": best_val_acc,
            "macro_f1": final_res["macro_f1"],
        }
