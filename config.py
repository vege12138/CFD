# =========================
#  W3 Configuration (OptInit Pattern)
# =========================
import os
import datetime
import argparse
import random
import numpy as np
import torch
import logging
import logging.config


class OptInit:
    """参数初始化器"""

    def __init__(self):
        parser = argparse.ArgumentParser(description='W3 Zero-Shot GNN')

        # 基础配置
        datasetLS = ["cora", "citeseer", "pubmed",  "bookhis","bookchild", "elephoto", "wikics",  ]
        parser.add_argument('--dataset', type=str, default=datasetLS[-1], help='Dataset name')
        parser.add_argument('--data_root', type=str, default='dataset', help='Data root directory')
        parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
        parser.add_argument('--seed', type=int, default=42, help='Random seed')
        parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
        parser.add_argument('--runs', type=int, default=5, help='Number of runs')

        # Co-Training超参数
        parser.add_argument('--warmup_epochs', type=int, default=110, help='Warmup epochs (W)')
        parser.add_argument('--label_update_interval', type=int, default=100 ,help='Label update interval (T)')
        parser.add_argument('--total_epochs', type=int, default=1600, help='Total training epochs')
        parser.add_argument('--co_train_lr', type=float, default=0.001, help='Co-training learning rate')
        parser.add_argument('--co_train_tau', type=float, default=0.1, help='Co-training temperature')
        parser.add_argument('--post_warmup_lr', type=float, default=0.00001, help='Learning rate after warmup')
        parser.add_argument('--post_warmup_tau', type=float, default=1.0, help='Temperature after warmup')

        # LP传播超参数
        parser.add_argument('--lp_alpha', type=float, default=0.6, help='LP propagation alpha')
        parser.add_argument('--lp_num_iter', type=int, default=1, help='LP propagation iterations')

        # GNN模型配置
        parser.add_argument('--gnn_model', type=str, default='SAGE', choices=['GCN', 'SAGE', 'MLP'],
                            help='GNN model type')
        parser.add_argument('--hidden_dim', type=int, default=1024, help='Hidden layer dimension')
        parser.add_argument('--num_layers', type=int, default=2, help='Number of GNN layers')
        # 训练配
        # LLM得分升级配置
        parser.add_argument('--num_conv_layers', type=int, default=2, help='Graph propagation layers for score upgrade')
        parser.add_argument('--m_ratio', type=float, default=0.05, help='High confidence node ratio')

        # LM配置
        parser.add_argument('--lm_model', type=str, default='intfloat/e5-base-v2', help='LM model name')
        parser.add_argument('--lm_batch_size', type=int, default=64, help='LM batch size')

        args = parser.parse_args()

        # 添加时间戳
        args.time = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")

        # 设备检查
        if args.device == 'cuda' and not torch.cuda.is_available():
            args.device = 'cpu'
            print("⚠️ CUDA not available, using CPU")

        self.args = args

    def initialize(self):
        """初始化：设置种子、日志、打印参数"""
        self.set_seed(self.args.seed)
        self.logging_init()
        self.print_args()
        return self.args

    def print_args(self):
        """格式化打印所有参数"""
        self.args.printer.info("=" * 60)
        self.args.printer.info("               W3 CONFIG")
        self.args.printer.info("=" * 60)
        self.args.printer.info(f"  Run Time: {self.args.time}")
        self.args.printer.info("-" * 60)

        # 将参数转换为字符串列表（排除printer）
        args_items = [(k, v) for k, v in self.args.__dict__.items() if k != 'printer']
        args_strs = [f"{k}: {v}" for k, v in args_items]

        # 计算最大宽度保证左对齐
        max_len = max(len(s) for s in args_strs) if args_strs else 0

        # 分组输出（每组2个）
        for i in range(0, len(args_strs), 2):
            group = [s.ljust(max_len) for s in args_strs[i:i + 2]]
            self.args.printer.info("  " + "    ".join(group))

        self.args.printer.info("=" * 60 + "\n")

    def logging_init(self):
        """初始化日志"""
        ERROR_FORMAT = "%(message)s"
        DEBUG_FORMAT = "%(message)s"
        LOG_CONFIG = {
            'version': 1,
            'formatters': {
                'error': {'format': ERROR_FORMAT},
                'debug': {'format': DEBUG_FORMAT}
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'formatter': 'debug',
                    'level': logging.DEBUG
                }
            },
            'root': {
                'handlers': ('console',),
                'level': 'DEBUG'
            }
        }
        logging.config.dictConfig(LOG_CONFIG)
        self.args.printer = logging.getLogger(__name__)

    def set_seed(self, seed=42):
        """固定随机种子"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def get_args():
    """获取初始化后的参数"""
    opt = OptInit()
    return opt.initialize()


if __name__ == '__main__':
    # 测试
    args = get_args()
    print(f"\nTest: dataset={args.dataset}, epochs={args.epochs}")
