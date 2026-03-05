# =========================
#  Step 2: Encode Embeddings
# =========================
"""
功能:
1. 使用LM编码raw_texts (TA嵌入)
2. 使用LM编码e_texts (E嵌入)
3. 使用LM编码label2class (原型嵌入)
4. 保存到geometric_data_with_texts.pt
"""
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import AutoTokenizer, AutoModel
from core.data_utils.load import load_data, save_data


class EmbeddingEncoder:
    """LM嵌入编码器"""
    
    def __init__(self, model_name="intfloat/e5-base-v2", device="cuda"):
        self.model_name = model_name
        self.device = device
        
        print(f"🔧 Initializing EmbeddingEncoder")
        print(f"   Model: {model_name}")
        print(f"   Device: {device}")
        
        # 加载模型
        print(f"📥 Loading model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        
        self.hidden_dim = self.model.config.hidden_size
        print(f"   Hidden dim: {self.hidden_dim}")
    
    def encode_texts(self, texts, batch_size=64, desc="Encoding"):
        """
        批量编码文本
        
        Args:
            texts: 文本列表
            batch_size: 批次大小
            desc: 进度条描述
            
        Returns:
            embeddings: [N, hidden_dim]
        """
        all_embs = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc=desc):
                batch_texts = texts[i:i + batch_size]
                
                # 处理空文本
                batch_texts = [t if t else " " for t in batch_texts]
                
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                ).to(self.device)
                
                outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
                emb = outputs.hidden_states[-1][:, 0, :]  # [CLS] token
                emb = F.normalize(emb, p=2, dim=1)
                
                all_embs.append(emb.cpu())
        
        return torch.cat(all_embs, dim=0)
    
    def encode_labels(self, label2class):
        """
        编码标签文本
        
        Args:
            label2class: {id: (name, description)}
            
        Returns:
            prototypes: [num_classes, hidden_dim]
        """
        label_texts = []
        for i in range(len(label2class)):
            if i in label2class:
                class_name, class_desc = label2class[i]
                label_text = f"{class_name}: {class_desc}"
                label_texts.append(label_text)
        
        print(f"📊 Encoding {len(label_texts)} label descriptions...")
        
        with torch.no_grad():
            inputs = self.tokenizer(
                label_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            ).to(self.device)
            
            outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
            label_emb = outputs.hidden_states[-1][:, 0, :]
            label_emb = F.normalize(label_emb, p=2, dim=1)
        
        return label_emb.cpu()


def encode_embeddings(dataset, data_root='dataset', model_name="intfloat/e5-base-v2", device="cuda"):
    """
    编码TA、E和原型嵌入
    
    Args:
        dataset: 数据集名称
        data_root: 数据根目录
        model_name: LM模型名称
        device: 运行设备
    """
    print(f"\n{'='*60}")
    print(f"Step 2: Encoding Embeddings for {dataset}")
    print(f"{'='*60}")

    # 加载数据
    data, num_classes = load_data(dataset, data_root)
    num_nodes = data.y.shape[0]

    # 获取文本
    ta_texts = data.raw_texts if hasattr(data, 'raw_texts') else []
    e_texts = data.e_texts if hasattr(data, 'e_texts') else [""] * num_nodes
    
    if len(ta_texts) != num_nodes:
        raise ValueError(f"TA texts count mismatch: {len(ta_texts)} vs {num_nodes}")
    
    print(f"   TA texts: {len(ta_texts)}")
    print(f"   E texts: {len(e_texts)}")

    # 初始化编码器
    encoder = EmbeddingEncoder(model_name, device)

    # 编码TA
    print(f"\n📝 Encoding TA texts...")
    ta_embeddings = encoder.encode_texts(ta_texts, desc="TA")

    # 编码E
    print(f"\n📝 Encoding E texts...")
    e_embeddings = encoder.encode_texts(e_texts, desc="E")

    # 编码标签原型
    print(f"\n📊 Encoding label prototypes...")
    label_prototypes = encoder.encode_labels(data.label2class)

    # 更新data对象
    data.ta_embeddings = ta_embeddings
    data.e_embeddings = e_embeddings
    data.label_prototypes = label_prototypes

    # 保存
    save_data(data, dataset, data_root)

    # 计算准确率
    ta_logits = ta_embeddings @ label_prototypes.T
    e_logits = e_embeddings @ label_prototypes.T
    
    ta_preds = ta_logits.argmax(dim=1)
    e_preds = e_logits.argmax(dim=1)
    
    labels = data.y.squeeze().cpu()
    ta_acc = (ta_preds == labels).float().mean().item()
    e_acc = (e_preds == labels).float().mean().item()

    print(f"\n{'='*60}")
    print(f"Encoding Summary")
    print(f"{'='*60}")
    print(f"   TA Embeddings: {ta_embeddings.shape}")
    print(f"   E Embeddings: {e_embeddings.shape}")
    print(f"   Label Prototypes: {label_prototypes.shape}")
    print(f"\n   TA Accuracy: {ta_acc:.4f} ({ta_acc*100:.2f}%)")
    print(f"   E Accuracy: {e_acc:.4f} ({e_acc*100:.2f}%)")
    print(f"{'='*60}\n")

    return data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Step 2: Encode Embeddings")
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--data_root", type=str, default="dataset")
    parser.add_argument("--model", type=str, default="intfloat/e5-base-v2")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    args.dataset = "wikics"
    
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"
    
    encode_embeddings(args.dataset, args.data_root, args.model, args.device)
