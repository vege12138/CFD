# =========================
#  Data Loading Utilities
# =========================
import os
import torch


def load_data(dataset, data_root='dataset'):
    """
    统一数据加载入口
    
    Args:
        dataset: 数据集名称 (如 'cora', 'citeseer', 'pubmed')
        data_root: 数据集根目录
        
    Returns:
        data: PyG Data对象，包含:
            - x: 节点特征 (如有)
            - edge_index: 边索引
            - y: 节点标签
            - raw_texts: TA原始文本
            - e_texts: E解释文本 (Step1后)
            - llm_score_matrix: LLM初始得分 (Step1后)
            - ta_embeddings: TA嵌入 (Step2后)
            - e_embeddings: E嵌入 (Step2后)
            - label_prototypes: 原型嵌入 (Step2后)
            - label2class: 标签ID到类别名映射
        num_classes: 类别数
    """
    # 构建文件路径
    data_dir = os.path.join(data_root, dataset)
    pt_file = os.path.join(data_dir, "geometric_data_with_texts.pt")

    print(f"📂 Loading dataset '{dataset}' from: {pt_file}")

    if not os.path.exists(pt_file):
        raise FileNotFoundError(f"Data file not found: {pt_file}")

    # 加载数据
    try:
        data = torch.load(pt_file, weights_only=False)
    except TypeError:
        data = torch.load(pt_file)

    # 解析类别数
    if hasattr(data, 'num_classes') and data.num_classes:
        num_classes = int(data.num_classes)
    elif hasattr(data, 'y') and data.y is not None:
        num_classes = int(data.y.max().item()) + 1
    else:
        class_map = {
            'cora': 7, 'citeseer': 6, 'pubmed': 3,
            'arxiv': 40, 'ogbn-arxiv': 40
        }
        num_classes = class_map.get(dataset.lower(), 0)

    print(f"   Nodes: {data.y.shape[0]}, Classes: {num_classes}")
    
    return data, num_classes


def save_data(data, dataset, data_root='dataset'):
    """保存数据到pt文件"""
    data_dir = os.path.join(data_root, dataset)
    os.makedirs(data_dir, exist_ok=True)
    
    pt_file = os.path.join(data_dir, "geometric_data_with_texts.pt")
    torch.save(data, pt_file)
    print(f"💾 Saved data to: {pt_file}")


def get_class_map(data):
    """
    从data.label2class构建类别名到ID的映射
    
    Args:
        data: 包含label2class属性的数据对象
        
    Returns:
        class_map: {类别名: 类别ID}
        
    注意: 
        - 对于带括号的类别名如 'cs.NA (Numerical Analysis)'，
          会同时添加完整名和简短名 'cs.NA' 两个映射
    """
    class_map = {}
    label2class = data.label2class

    if isinstance(label2class, dict):
        for idx, val in label2class.items():
            class_name = val[0] if isinstance(val, (list, tuple)) else val
            full_name = str(class_name).strip()
            class_map[full_name] = idx
            
            # 如果类别名包含括号，也添加简短名映射
            if '(' in full_name:
                short_name = full_name.split('(')[0].strip()
                class_map[short_name] = idx
                
    elif isinstance(label2class, list):
        for idx, class_name in enumerate(label2class):
            full_name = str(class_name).strip()
            class_map[full_name] = idx
            
            # 如果类别名包含括号，也添加简短名映射
            if '(' in full_name:
                short_name = full_name.split('(')[0].strip()
                class_map[short_name] = idx

    return class_map
