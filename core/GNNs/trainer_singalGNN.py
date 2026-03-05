# =========================
#  Single-GNN Trainer (Ablation)
# =========================
"""
Ablation:
1) Single GNN
2) No edge masking (use original edge_index)
3) Train with LP hard cross-entropy (all nodes)
4) Evaluate every 10 epochs on all nodes
5) Final acc = accuracy after finishing training
6) lr/tau/total_epochs are loaded from dataset_params_singalGNN.json in the same folder
"""

import os
import json
import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score


LOG_FREQ = 10


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class GNNTrainer:
    """Single GNN trainer with LP hard cross-entropy supervision"""

    def __init__(self, args, data, num_classes: int):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = data.to(self.device)
        self.num_classes = num_classes
        self.num_nodes = self.data.y.size(0)

        set_seed(getattr(args, "seed", 0))

        # model hyperparams (keep only those needed for model construction)
        self.hidden_dim = getattr(args, "hidden_dim", 256)
        self.num_layers = getattr(args, "num_layers", 2)
        self.dropout = getattr(args, "dropout", 0.5)

        # load training hyperparams from json (same folder as this file)
        dataset_name = getattr(args, "dataset", None)
        self._load_train_params_from_json(dataset_name)

        # features / prototypes
        self.ta_features = self.data.ta_embeddings.to(self.device)          # [N, d]
        self.label_prototypes = self.data.label_prototypes.to(self.device)  # [C, d]
        assert self.label_prototypes.size(0) == self.num_classes

        # LP distribution -> hard labels
        self.P_distribution = self._load_lp_distribution(dataset_name)      # [N, C]
        self.P_distribution = self._row_normalize(self.P_distribution)
        self.lp_hard_labels = self.P_distribution.argmax(dim=1).long()      # [N]

        # sanity check
        y = self.data.y.squeeze()
        init_acc = (self.lp_hard_labels == y).float().mean().item()
        print(f"📊 LP hard-label Accuracy: {init_acc:.4f} ({init_acc * 100:.2f}%)")

        # model / optimizer
        self._build_model()
        self.optimizer = torch.optim.Adam(self.gnn.parameters(), lr=self.lr)

    # ---------------------------
    # utils
    # ---------------------------
    def _row_normalize(self, mat: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        return mat / mat.sum(dim=1, keepdim=True).clamp_min(eps)

    def _load_train_params_from_json(self, dataset_name: str):
        if dataset_name is None:
            raise ValueError("args.dataset is required for selecting params in dataset_params_singalGNN.json")

        json_path = os.path.join(os.path.dirname(__file__), "dataset_params_singalGNN.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Not found: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        ds_cfg = cfg.get(str(dataset_name), cfg.get("default", None))
        if ds_cfg is None:
            raise KeyError(f"Cannot find dataset '{dataset_name}' or 'default' in {json_path}")

        # only keep these three
        self.lr = float(ds_cfg["lr"])
        self.tau = float(ds_cfg["tau"])
        self.total_epochs = int(ds_cfg["total_epochs"])

        print(f"⚙️ Loaded params from dataset_params_singalGNN.json: "
              f"dataset={dataset_name}, lr={self.lr}, tau={self.tau}, total_epochs={self.total_epochs}")

    # ---------------------------
    # load/build
    # ---------------------------
    def _load_lp_distribution(self, dataset_name: str):
        if dataset_name is None:
            raise ValueError("args.dataset is required to locate lp_best_distribution.pt")

        data_dir = os.path.join("dataset", str(dataset_name))
        lp_path = os.path.join(data_dir, "lp_best_distribution.pt")
        if not os.path.exists(lp_path):
            raise FileNotFoundError(f"Not found: {lp_path}")

        lp_dist = torch.load(lp_path, map_location=self.device)
        return lp_dist.to(self.device)

    def _build_model(self):
        model_name = getattr(self.args, "gnn_model", "GCN")
        if model_name == "GCN":
            from core.GNNs.GCN.model import GCN as GNN
        elif model_name == "SAGE":
            from core.GNNs.SAGE.model import SAGE as GNN
        else:
            from core.GNNs.MLP.model import MLP as GNN

        in_dim = self.ta_features.size(1)
        out_dim = self.label_prototypes.size(1)

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
    # forward / eval
    # ---------------------------
    def _forward_logits(self) -> torch.Tensor:
        x = self.ta_features
        edge_index = self.data.edge_index

        try:
            z = self.gnn(x, edge_index)
        except TypeError:
            z = self.gnn(x)

        z = F.normalize(z, dim=1)
        proto = F.normalize(self.label_prototypes, dim=1)
        logits = (z @ proto.t()) / float(self.tau)
        return logits

    @torch.no_grad()
    def evaluate(self):
        self.gnn.eval()
        y = self.data.y.squeeze()

        logits = self._forward_logits()
        preds = logits.argmax(dim=1)

        acc = (preds == y).float().mean().item()
        macro = f1_score(y.detach().cpu().numpy(), preds.detach().cpu().numpy(), average="macro")
        return {"acc": acc, "macro_f1": macro}

    # ---------------------------
    # train
    # ---------------------------
    def train(self):
        y = self.data.y.squeeze()
        p_acc = (self.lp_hard_labels == y).float().mean().item()

        print(f"{'='*60}")
        print(f"🚀 Ablation Train: Single-GNN, LP hard cross-entropy, all-node train/eval")
        print(f"   epochs={self.total_epochs}, lr={self.lr}, tau={self.tau}")
        print(f"   LP hard-label Acc={p_acc:.4f}")
        print(f"{'='*60}")

        best_acc = -1.0

        for epoch in range(1, self.total_epochs + 1):
            self.gnn.train()
            self.optimizer.zero_grad()

            logits = self._forward_logits()
            loss = F.cross_entropy(logits, self.lp_hard_labels)

            loss.backward()
            self.optimizer.step()

            if epoch % LOG_FREQ == 0 or epoch == 1:
                res = self.evaluate()
                best_acc = max(best_acc, res["acc"])
                print(f"[Epoch {epoch:3d}] loss={loss.item():.4f} | "
                      f"Acc={res['acc']:.4f} MacroF1={res['macro_f1']:.4f} | best_acc={best_acc:.4f}")

        final_res = self.evaluate()

        print(f"{'='*60}")
        print(f"✅ Done. BestAcc(during train)={best_acc:.4f} | "
              f"FinalAcc(after train)={final_res['acc']:.4f} MacroF1={final_res['macro_f1']:.4f}")
        print(f"{'='*60}")

        return {
            "best_acc": best_acc,
            "acc": final_res["acc"],
            "macro_f1": final_res["macro_f1"],
        }
