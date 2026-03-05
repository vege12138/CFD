import os

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.manifold import TSNE
import random
import numpy as np
import matplotlib.pyplot as plt


LOG_FREQ = 10


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def visualize_tsne(dataset: str, embeddings: torch.Tensor, labels: torch.Tensor, epoch: int, tag: str):
    """
    使用 t-SNE 可视化节点嵌入并保存为 PDF/PNG
    - dataset: 数据集名称
    - embeddings: [N, D] 节点嵌入
    - labels: [N] 或 [N, 1] 节点标签
    - epoch: 当前 epoch
    - tag: 阶段标记，比如 'input', 'warmup', 'final'
    """
    os.makedirs("tSNE", exist_ok=True)

    emb_np = embeddings.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    if labels_np.ndim > 1:
        labels_np = labels_np.squeeze()

    tsne = TSNE(n_components=2, init="pca", random_state=0)
    emb_2d = tsne.fit_transform(emb_np)

    plt.figure(figsize=(8, 8))
    plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels_np, cmap="tab10", s=10, alpha=0.7)
    plt.title(f"t-SNE Visualization - {tag} - Epoch {epoch}")
    plt.xticks([])
    plt.yticks([])

    folder_path = f"visualization/tSNE_{dataset}/{tag}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)

    pdf_path = os.path.join(folder_path, f"epoch_{epoch}.pdf")
    png_path = os.path.join(folder_path, f"epoch_{epoch}.png")
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.savefig(png_path, format="png", bbox_inches="tight")
    plt.close()


class GNNTrainer:
    """
    带 t-SNE 可视化的交替式 GNN-LP 训练器:
    - 在模型训练前可视化原始 TA 节点嵌入
    - 在 warmup 阶段每隔 10 个 epoch 可视化 GNN 输出嵌入
    - 在 warmup 之后的最终训练阶段每隔 10 个 epoch 可视化 GNN 输出嵌入
    """

    def __init__(self, args, data, num_classes):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = data.to(self.device)
        self.num_classes = num_classes
        self.num_nodes = data.y.size(0)
        set_seed(args.seed)

        # 超参数
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.dropout = args.dropout

        # Co-Training 超参数
        self.W = args.warmup_epochs  # 预热 epoch 数
        self.T = args.label_update_interval  # 标签更新间隔
        self.total_epochs = args.total_epochs  # 总训练 epoch 数
        self.co_train_lr = args.co_train_lr  # Co-training 学习率
        self.co_train_tau = args.co_train_tau  # Co-training 温度系数
        self.post_warmup_lr = args.post_warmup_lr  # 预热后学习率
        self.post_warmup_tau = args.post_warmup_tau  # 预热后温度系数

        # LP 传播超参数
        self.lp_alpha = args.lp_alpha  # LP 传播 alpha
        self.lp_num_iter = args.lp_num_iter  # LP 传播迭代次数

        # 加载 TA 嵌入和标签原型
        self.ta_features = data.ta_embeddings.to(self.device)
        self.label_prototypes = data.label_prototypes.to(self.device)

        # 加载 LP 预热分布
        self.P_distribution = self._load_lp_distribution(args.dataset)

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
        print(f"📊 Alternating GNN-LP Trainer with t-SNE Initialized")
        print(f"   Nodes: {self.num_nodes}, Classes: {self.num_classes}")
        print(f"   Mode: GNN training ↔ LP propagation alternating + t-SNE visualization")
        print(f"{'=' * 60}\n")

        # 在训练开始前可视化输入模型前的节点嵌入
        visualize_tsne(
            dataset=args.dataset,
            embeddings=self.ta_features,
            labels=self.data.y,
            epoch=0,
            tag="input",
        )

    def _load_lp_distribution(self, dataset_name):
        """加载 LP 预热分布"""
        data_dir = os.path.join("dataset", dataset_name)
        lp_path = os.path.join(data_dir, "lp_best_distribution.pt")
        lp_dist = torch.load(lp_path).to(self.device)
        return lp_dist

    def _build_model(self):
        """构建双 GNN 模型"""
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
            use_pred=False,
        ).to(self.device)

        self.gnn2 = GNN(
            in_channels=input_dim,
            hidden_channels=self.hidden_dim,
            out_channels=output_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            use_pred=False,
        ).to(self.device)

        print(f"✅ GNN1: {model_name} (768 → 768)")
        print(f"✅ GNN2: {model_name} (768 → 768)")

    def _edge_masking(self, edge_index, drop_ratio=0.1):
        """随机遮蔽边"""
        num_edges = edge_index.size(1)
        edge_mask = torch.bernoulli(
            torch.ones(num_edges, device=self.device) * (1 - drop_ratio)
        ).bool()
        edge_index_masked = edge_index[:, edge_mask]
        return edge_index_masked

    @torch.no_grad()
    def _label_propagation(self, Y0, alpha=0.7, num_iter=3):
        """标签传播算法"""
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
            Y_new = alpha * torch.sparse.mm(S, Y) + (1 - alpha) * Y0
            Y = Y_new

        Y = Y / Y.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return Y

    def train(self):
        """协同训练并在训练过程中进行 t-SNE 可视化"""
        # ==================== 超参数 ====================
        W = self.W  # 预热 epoch 数
        T = self.T  # 标签更新间隔
        total_epochs = self.total_epochs  # 总训练 epoch 数
        lr = self.co_train_lr  # Co-training 学习率
        tau = self.co_train_tau  # Co-training 温度系数

        optimizer1 = torch.optim.Adam(self.gnn1.parameters(), lr=lr)
        optimizer2 = torch.optim.Adam(self.gnn2.parameters(), lr=lr)
        print(f"{'=' * 60}")
        print(f"🚀 Co-Training with AB Split (t-SNE Enabled)")
        print(f"   Warmup W={W}, Update Interval T={T}, Total={total_epochs}")
        print(f"{'=' * 60}")

        y = self.data.y.squeeze()
        edge_index = self.data.edge_index

        # ==================== 随机划分数据集为 A 和 B ====================
        n = self.num_nodes
        perm = torch.randperm(n, device=self.device)
        split = n // 2
        mask_A = torch.zeros(n, dtype=torch.bool, device=self.device)
        mask_B = torch.zeros(n, dtype=torch.bool, device=self.device)
        mask_A[perm[:split]] = True
        mask_B[perm[split:]] = True

        print(f"\n   📌 Data Split: A={mask_A.sum().item()}, B={mask_B.sum().item()}")

        # 初始化训练标签：使用 P 分布的 argmax
        labels_A = self.P_distribution.argmax(dim=1).clone()  # GNN1 在 A 上的训练标签
        labels_B = self.P_distribution.argmax(dim=1).clone()  # GNN2 在 B 上的训练标签

        # 打印初始 P 分布准确率
        p_acc_A = (labels_A[mask_A] == y[mask_A]).float().mean().item()
        p_acc_B = (labels_B[mask_B] == y[mask_B]).float().mean().item()
        p_acc_total = (self.P_distribution.argmax(dim=1) == y).float().mean().item()
        print(
            f"   Initial P_dist Acc: Total={p_acc_total:.4f} | A={p_acc_A:.4f} | B={p_acc_B:.4f}"
        )

        best_acc = 0.0
        best_preds = None

        # W 后保留最高准确率的标签
        best_g1_A_acc = 0.0
        best_g1_A_preds = None  # GNN1 在 A 上的最佳预测
        best_g2_B_acc = 0.0
        best_g2_B_preds = None  # GNN2 在 B 上的最佳预测

        dataset_name = self.args.dataset

        # ==================== 主训练循环 ====================
        for epoch in range(total_epochs):
            self.gnn1.train()
            self.gnn2.train()

            # 边 dropout
            edge_index_dropped1 = self._edge_masking(edge_index, drop_ratio=0.2)
            edge_index_dropped2 = self._edge_masking(edge_index, drop_ratio=0.2)

            # ========== GNN1 前向传播 ==========
            optimizer1.zero_grad()
            z1 = self.gnn1(self.ta_features, edge_index_dropped1)
            z1 = F.normalize(z1, dim=1)
            proto_norm = F.normalize(self.label_prototypes, dim=1)
            sim1 = torch.mm(z1, proto_norm.T) / tau
            pred1 = F.softmax(sim1, dim=1)

            # ========== GNN2 前向传播 ==========
            optimizer2.zero_grad()
            z2 = self.gnn2(self.ta_features, edge_index_dropped2)
            z2 = F.normalize(z2, dim=1)
            sim2 = torch.mm(z2, proto_norm.T) / tau
            pred2 = F.softmax(sim2, dim=1)

            # ========== 计算损失 ==========
            # GNN1 用 A 训练，GNN2 用 B 训练
            loss1_ce = F.cross_entropy(sim1[mask_A], labels_A[mask_A])
            loss2_ce = F.cross_entropy(sim2[mask_B], labels_B[mask_B])

            loss1 = loss1_ce
            loss2 = loss2_ce

            loss1.backward(retain_graph=True)
            loss2.backward()

            optimizer1.step()
            optimizer2.step()

            # ========== t-SNE 可视化：根据阶段和间隔保存 ==========
            if epoch % 5 == 0 and epoch <= W:
                with torch.no_grad():
                    # 使用两个 GNN 输出的平均作为当前节点嵌入
                    z_mean = (z1 + z2) / 2.0
                    phase_tag = "warmup" if epoch < W else "final"
                    visualize_tsne(
                        dataset=dataset_name,
                        embeddings=z_mean,
                        labels=self.data.y,
                        epoch=epoch,
                        tag=phase_tag,
                    )
            elif epoch % 60 == 0:
                with torch.no_grad():
                    # 使用两个 GNN 输出的平均作为当前节点嵌入
                    z_mean = (z1 + z2) / 2.0
                    phase_tag = "warmup" if epoch < W else "final"
                    visualize_tsne(
                        dataset=dataset_name,
                        embeddings=z_mean,
                        labels=self.data.y,
                        epoch=epoch,
                        tag=phase_tag,
                    )


            # ========== 预热后每隔 T 代更新标签 ==========
            if epoch >= W and (epoch - W) % T == 0:
                lr = self.post_warmup_lr
                tau = self.post_warmup_tau
                optimizer1 = torch.optim.Adam(self.gnn1.parameters(), lr=lr)
                optimizer2 = torch.optim.Adam(self.gnn2.parameters(), lr=lr)
                with torch.no_grad():
                    pred1_labels = pred1.argmax(dim=1)
                    pred2_labels = pred2.argmax(dim=1)

                    pred_labels = ((pred1 + pred2) / 2).argmax(dim=1)
                    labels_B[mask_B] = pred_labels[mask_B]
                    labels_A[mask_A] = pred_labels[mask_A]

                    new_acc_A = (labels_A[mask_A] == y[mask_A]).float().mean().item()
                    new_acc_B = (labels_B[mask_B] == y[mask_B]).float().mean().item()
                    print(
                        f"   [Epoch {epoch}] Labels updated! Label Acc: A={new_acc_A:.4f} B={new_acc_B:.4f}"
                    )

            # ========== 第 W 代特殊评估 ==========
            if epoch == W:
                with torch.no_grad():
                    pred_mean = (pred1 + pred2) / 2
                    pred_mean_acc = (
                        (pred_mean.argmax(dim=1) == y).float().mean().item()
                    )

                    print(f"\n   📌 Warmup Complete (Epoch {W}):")
                    print(f"      pred_mean Acc： {pred_mean_acc:.4f}")

                    print(f"\n      🔗 Fusing pred_mean (argmax=1) + P_distribution:")
                    pred_mean_argmax = pred_mean.argmax(dim=1)
                    pred_mean_onehot = torch.zeros_like(pred_mean)
                    pred_mean_onehot.scatter_(1, pred_mean_argmax.unsqueeze(1), 1.0)

                    fused_dist = pred_mean_onehot + self.P_distribution
                    fused_dist_acc = (
                        fused_dist.argmax(dim=1) == y
                    ).float().mean().item()
                    print(
                        f"      Fused Dist Acc (before top-k): {fused_dist_acc:.4f}"
                    )

                    for top_k in [1, 2, 3]:
                        fused_topk = fused_dist.clone()
                        topk_values, topk_indices = torch.topk(
                            fused_topk, k=top_k, dim=1
                        )

                        mask_topk = torch.zeros_like(fused_topk)
                        mask_topk.scatter_(1, topk_indices, 1.0)
                        fused_topk = fused_topk * mask_topk

                        fused_topk = fused_topk / fused_topk.sum(
                            dim=1, keepdim=True
                        ).clamp_min(1e-12)

                        fused_topk_lp = self._label_propagation(
                            fused_topk, alpha=self.lp_alpha, num_iter=self.lp_num_iter
                        )
                        fused_topk_lp_acc = (
                            fused_topk_lp.argmax(dim=1) == y
                        ).float().mean().item()

                        print(
                            f"      Fused Dist Acc (top-{top_k} + LP): {fused_topk_lp_acc:.4f} (Delta: {fused_topk_lp_acc - pred_mean_acc:+.4f})"
                        )

                    merged_preds = torch.zeros_like(y)
                    merged_preds[mask_A] = pred1.argmax(dim=1)[mask_A]
                    merged_preds[mask_B] = pred2.argmax(dim=1)[mask_B]
                    merged_acc_warmup = (merged_preds == y).float().mean().item()

                    print(f"      Merged (G1-A + G2-B) Acc: {merged_acc_warmup:.4f}")

                    print(f"      🔄 Swapping partitions: GNN1->B, GNN2->A")
                    mask_A, mask_B = mask_B, mask_A
                    print(
                        f"      New split: GNN1 trains on B={mask_B.sum().item()}, GNN2 trains on A={mask_A.sum().item()}\n"
                    )

            # ========== 评估 ==========
            if epoch % 10 == 0 or epoch == total_epochs - 1:
                with torch.no_grad():
                    pred_mean = (pred1 + pred2) / 2
                    preds = pred_mean.argmax(dim=1)
                    acc = (preds == y).float().mean().item()

                    pred1_acc_B = (
                        pred1.argmax(dim=1)[mask_B] == y[mask_B]
                    ).float().mean().item()
                    pred2_acc_A = (
                        pred2.argmax(dim=1)[mask_A] == y[mask_A]
                    ).float().mean().item()

                    pred1_acc_A = (
                        pred1.argmax(dim=1)[mask_A] == y[mask_A]
                    ).float().mean().item()
                    pred2_acc_B = (
                        pred2.argmax(dim=1)[mask_B] == y[mask_B]
                    ).float().mean().item()

                    merged_preds_eval = torch.zeros_like(y)
                    merged_preds_eval[mask_A] = pred1.argmax(dim=1)[mask_A]
                    merged_preds_eval[mask_B] = pred2.argmax(dim=1)[mask_B]
                    merged_acc_eval = (
                        merged_preds_eval == y
                    ).float().mean().item()

                    if acc > best_acc:
                        best_acc = acc
                        best_preds = preds.clone()

                    if epoch >= W:
                        if pred1_acc_A > best_g1_A_acc:
                            best_g1_A_acc = pred1_acc_A
                            best_g1_A_preds = pred1.argmax(dim=1)[
                                mask_A
                            ].clone()
                        if pred2_acc_B > best_g2_B_acc:
                            best_g2_B_acc = pred2_acc_B
                            best_g2_B_preds = pred2.argmax(dim=1)[
                                mask_B
                            ].clone()

                    phase = "Warmup" if epoch < W else "Change"
                    print(
                        f"   [{phase}] Epoch {epoch:3d} | Loss1 {loss1.item():.4f} Loss2 {loss2.item():.4f} | "
                        f"Acc {acc:.4f} | G1(A/B) {pred1_acc_A:.4f}/{pred1_acc_B:.4f} | G2(A/B) {pred2_acc_A:.4f}/{pred2_acc_B:.4f}"
                    )

        # ==================== 最终评估 ====================
        self.gnn1.eval()
        self.gnn2.eval()
        with torch.no_grad():
            z1_final = self.gnn1(self.ta_features, edge_index)
            z2_final = self.gnn2(self.ta_features, edge_index)
            z1_final = F.normalize(z1_final, dim=1)
            z2_final = F.normalize(z2_final, dim=1)
            z_final_mean = (z1_final + z2_final) / 2.0

            # 最终一次 t-SNE 可视化（phase 记为 final_last）
            visualize_tsne(
                dataset=dataset_name,
                embeddings=z_final_mean,
                labels=self.data.y,
                epoch=total_epochs,
                tag="final_last",
            )

            proto_norm = F.normalize(self.label_prototypes, dim=1)
            sim1_final = torch.mm(z1_final, proto_norm.T)
            sim2_final = torch.mm(z2_final, proto_norm.T)

            pred1_final = F.softmax(sim1_final, dim=1)
            pred2_final = F.softmax(sim2_final, dim=1)
            pred_mean_final = (pred1_final + pred2_final) / 2

            y_full = self.data.y.squeeze()
            preds_full = pred_mean_final.argmax(dim=1)
            final_acc = (preds_full == y_full).float().mean().item()

        final_preds = preds_full
        final_pred1 = pred1_final
        final_pred2 = pred2_final

        final_merged_preds = torch.zeros_like(y_full)
        final_merged_preds[mask_A] = final_pred1.argmax(dim=1)[mask_A]
        final_merged_preds[mask_B] = final_pred2.argmax(dim=1)[mask_B]
        final_merged_acc = (final_merged_preds == y_full).float().mean().item()

        if final_acc > best_acc:
            best_acc = final_acc
            best_preds = final_preds.clone()

        if final_merged_acc > best_acc:
            best_acc = final_merged_acc
            best_preds = final_merged_preds.clone()

        final_f1 = f1_score(
            y_full.cpu().numpy(), best_preds.cpu().numpy(), average="macro"
        )

        if best_g1_A_preds is not None and best_g2_B_preds is not None:
            best_merged_preds = torch.zeros_like(y_full)
            best_merged_preds[mask_A] = best_g1_A_preds
            best_merged_preds[mask_B] = best_g2_B_preds
            best_merged_acc = (best_merged_preds == y_full).float().mean().item()
        else:
            best_merged_acc = 0.0

        print(f"\n{'=' * 60}")
        print(f"✅ Co-Training Complete with t-SNE!")
        print(f"   P_dist Acc: {p_acc_total:.4f}")
        print(f"   Best Acc:   {best_acc:.4f} (Delta: {best_acc - p_acc_total:+.4f})")
        print(f"   Final Acc:  {final_acc:.4f}")
        print(
            f"   Final Merged (G1-A + G2-B) Acc: {final_merged_acc:.4f}"
        )
        print(
            f"   Best G1-A({best_g1_A_acc:.4f}) + G2-B({best_g2_B_acc:.4f}) Merged Acc: {best_merged_acc:.4f}"
        )
        print(f"   Macro F1:   {final_f1:.4f}")
        print(f"{'=' * 60}")

        return {"best_acc": best_acc, "acc": best_acc, "macro_f1": final_f1}


