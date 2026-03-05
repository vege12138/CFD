# =========================
#  Step 1: Process LLM JSON
# =========================
"""
功能:
1. 解析LLM生成的JSON文件，提取解释文本(e_text)和得分
2. 构建初始LLM得分矩阵
3. 保存到geometric_data_with_texts.pt
"""
import os
import json
import re
import torch
import pandas as pd
from tqdm import tqdm

from core.data_utils.load import load_data, save_data, get_class_map


def parse_llm_json_string(json_str):
    """
    解析LLM返回的JSON字符串
    
    包含功能：
    1. 去除 <think> 思考链
    2. 修复非标准转义符
    3. 提取Markdown代码块
    """
    if not json_str:
        return []

    # 去除 <think> 思考过程
    if "</think>" in json_str:
        json_str = json_str.split("</think>")[-1].strip()

    # 修复单引号转义
    json_str = json_str.replace(r"\'", "'")

    # 修复非法转义符
    pattern = r'\\(?![/\"\\\bfnrtu])'
    json_str = re.sub(pattern, r"\\\\", json_str)

    try:
        return json.loads(json_str, strict=False)
    except json.JSONDecodeError:
        pass

    # 尝试提取Markdown代码块
    match = re.search(r"```json\s*(\[.*?\])\s*```", json_str, re.DOTALL)
    if not match:
        match = re.search(r"```\s*(\[.*?\])\s*```", json_str, re.DOTALL)

    if match:
        try:
            txt = match.group(1)
            txt = txt.replace(r"\'", "'")
            txt = re.sub(pattern, r"\\\\", txt)
            return json.loads(txt, strict=False)
        except:
            pass

    # 暴力提取 [...]
    try:
        start_idx = json_str.find('[')
        end_idx = json_str.rfind(']')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            candidate = json_str[start_idx: end_idx + 1]
            candidate = candidate.replace(r"\'", "'")
            candidate = re.sub(pattern, r"\\\\", candidate)
            return json.loads(candidate, strict=False)
    except:
        pass

    return []


def get_llm_score_matrix(processed_list, class_map, num_nodes, num_classes):
    """
    从处理后的数据构建LLM得分矩阵
    
    Args:
        processed_list: 包含score字典的列表
        class_map: {类别名: 类别ID}
        num_nodes: 节点数
        num_classes: 类别数
        
    Returns:
        score_matrix: [num_nodes, num_classes]
    """
    score_matrix = torch.zeros(num_nodes, num_classes)

    for item in processed_list:
        node_id = item.get("node_id", 0)
        scores = item.get("score", {})

        if scores:
            for class_name, score_value in scores.items():
                class_name = str(class_name).strip()
                if class_name in class_map:
                    class_id = class_map[class_name]
                    score_matrix[node_id, class_id] = score_value / 100.0

    return score_matrix


def process_llm_json(dataset, json_dir, data_root='dataset', save_csv=True):
    """
    处理LLM JSON文件，提取e_text和得分矩阵
    
    Args:
        dataset: 数据集名称
        json_dir: LLM JSON文件目录
        data_root: 数据根目录
        save_csv: 是否保存CSV文件
    """
    print(f"\n{'='*60}")
    print(f"Step 1: Processing LLM JSON for {dataset}")
    print(f"{'='*60}")

    # 加载原始数据
    data, num_classes = load_data(dataset, data_root)
    num_nodes = data.y.shape[0]
    
    # 获取类别映射
    class_map = get_class_map(data)
    print(f"   Class map: {class_map}")

    # 处理JSON文件
    processed_list = []
    e_texts = []
    parsing_errors = []
    missed_files = []  # 新增：记录缺失的文件

    if not os.path.exists(json_dir):
        raise FileNotFoundError(f"JSON directory not found: {json_dir}")

    print(f"   Processing {num_nodes} nodes from {json_dir}...")

    for node_idx in tqdm(range(num_nodes), desc="Processing"):
        json_filename = f"{node_idx}.json"
        file_path = os.path.join(json_dir, json_filename)

        if not os.path.exists(file_path):
            # 节点文件不存在，使用空值
            missed_files.append(node_idx)
            e_texts.append("")
            processed_list.append({
                "node_id": node_idx,
                "score": {},
                "e_text": ""
            })
            continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = json.load(f)

            llm_output_str = file_content.get("answer", "")
            predictions = parse_llm_json_string(llm_output_str)

            explanation = ""
            score_dict = {}

            if predictions and isinstance(predictions, list):
                if len(predictions) > 0:
                    top1_pred = predictions[0]
                    explanation = top1_pred.get("explanation", "").strip()

                for pred in predictions:
                    ans = pred.get("answer", "unknown").strip()
                    try:
                        conf = float(pred.get("confidence", 0))
                    except:
                        conf = 0
                    score_dict[ans] = conf
            else:
                parsing_errors.append(node_idx)

            e_texts.append(explanation)
            processed_list.append({
                "node_id": node_idx,
                "score": score_dict,
                "e_text": explanation
            })

        except Exception as e:
            print(f"Error at node {node_idx}: {e}")
            parsing_errors.append(node_idx)
            e_texts.append("")
            processed_list.append({
                "node_id": node_idx,
                "score": {},
                "e_text": ""
            })

    # 构建LLM得分矩阵
    print("\n   Building LLM score matrix...")
    llm_score_matrix = get_llm_score_matrix(processed_list, class_map, num_nodes, num_classes)

    # 更新data对象
    data.e_texts = e_texts
    data.llm_score_matrix = llm_score_matrix

    # 保存更新后的data
    save_data(data, dataset, data_root)

    # 保存CSV (可选)
    if save_csv:
        csv_path = os.path.join(data_root, dataset, f"{dataset}_processed_with_explanation.csv")
        df = pd.DataFrame(processed_list)
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"💾 Saved CSV to: {csv_path}")

    # 打印统计
    print(f"\n{'='*60}")
    print(f"Processing Summary")
    print(f"{'='*60}")
    print(f"   Total Nodes: {num_nodes}")
    print(f"   Successfully Processed: {len(processed_list) - len(missed_files) - len(parsing_errors)}")
    print(f"   ❌ Missing JSON Files: {len(missed_files)}")
    print(f"   ⚠️ Parsing Errors: {len(parsing_errors)}")
    print(parsing_errors)
    
    if len(missed_files) > 0 and len(missed_files) <= 10:
        print(f"      Missing IDs: {missed_files}")
    elif len(missed_files) > 10:
        print(f"      Missing IDs (first 10): {missed_files[:10]}...")
    
    if len(parsing_errors) > 0 and len(parsing_errors) <= 10:
        print(f"      Error IDs: {parsing_errors}")
    elif len(parsing_errors) > 10:
        print(f"      Error IDs (first 10): {parsing_errors[:10]}...")
    
    # 计算LLM准确率
    llm_preds = llm_score_matrix.argmax(dim=1)
    llm_acc = (llm_preds == data.y.squeeze()).float().mean().item()
    print(f"\n   🎯 LLM Initial Accuracy: {llm_acc:.4f} ({llm_acc*100:.2f}%)")
    print(f"{'='*60}\n")

    return data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Step 1: Process LLM JSON")
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--json_dir", type=str, help="LLM JSON files directory")
    parser.add_argument("--data_root", type=str, default="dataset")
    
    args = parser.parse_args()
    args.dataset= "pubmed"

    args.json_dir= rf"E:\CS\research\w3\w3_code_project\dataset\dataset\arxiv\vllm\json"
    args.json_dir= rf"dataset\{args.dataset}"

    process_llm_json(args.dataset, args.json_dir, args.data_root)
