import pandas as pd
import os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# 指定使用的 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 选择 GPU 0，修改为需要的 GPU 索引

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 定义自定义数据集
class ProteinDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences.iloc[idx]

# 加载 CSV 文件并过滤数据
file_path = './data/brenda/filtered_ogt_with_sequences.csv'
ogt_df = pd.read_csv(file_path)

# 过滤掉缺失值并筛选序列长度 > 50 的行
ogt_df_clean = ogt_df.dropna(subset=['uniprot_id', 'ogt', 'ogt_range', 'protein_sequence'])
ogt_df_filtered = ogt_df_clean[ogt_df_clean['protein_sequence'].apply(len) > 50]
print(f"过滤后的样本总数: {ogt_df_filtered.shape[0]}")

# 更新 X 和 y
X = ogt_df_filtered['protein_sequence']
y = ogt_df_filtered['ogt']

# 加载本地模型
model_path = './esm2'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)
model = model.to(device).eval()  # 将模型移动到 GPU，并设置为评估模式

# 定义嵌入生成函数
def generate_embeddings(sequences, model, tokenizer, device):
    tokens = tokenizer(sequences, return_tensors='pt', padding=True, truncation=True, max_length=512)
    tokens = {key: value.to(device) for key, value in tokens.items()}
    with torch.no_grad():
        embedding = model(**tokens)
    # 提取 [CLS] token 或平均池化嵌入
    cls_embedding = embedding.last_hidden_state[:, 0, :]
    return cls_embedding

# 创建数据集和数据加载器
dataset = ProteinDataset(X)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

# 生成嵌入并保存到列表
X_embeddings = []
print("开始生成嵌入...")
for batch in tqdm(dataloader):
    batch_embeddings = generate_embeddings(batch, model, tokenizer, device)
    X_embeddings.append(batch_embeddings.cpu())  # 移动到 CPU 节省 GPU 内存

# 将所有嵌入拼接成一个张量
X_embeddings = torch.cat(X_embeddings)

# 保存嵌入和标签
output_dir = './output'
os.makedirs(output_dir, exist_ok=True)

# 保存嵌入为 .pt 文件
embedding_file = os.path.join(output_dir, 'protein_embeddings.pt')
torch.save(X_embeddings, embedding_file)
print(f"嵌入已保存到 {embedding_file}")

# 保存标签为 .csv 文件
labels_file = os.path.join(output_dir, 'protein_labels.csv')
y.to_csv(labels_file, index=False)
print(f"标签已保存到 {labels_file}")
