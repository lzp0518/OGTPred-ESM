import os
import pandas as pd

# 已成功提取的特征文件列表
tsv_files = [
    'AAC_features.tsv',
    'CKSAAP_features.tsv',
    'DPC_features.tsv',
    'PSSM_features.tsv',
    'CTDC_features.tsv',
    'CTDT_features.tsv',
    'CTDD_features.tsv',
    'QSOrder_features.tsv',
    'PAAC_features.tsv'
]

summary_data = []

print("✅ 已提取特征及其维度/样本数：\n")
print(f"{'特征类型':<10} {'样本数':<10} {'维度':<10}")
print("-" * 32)

for file in tsv_files:
    if os.path.exists(file):
        df = pd.read_csv(file, sep='\t')
        feature_type = file.split('_')[0]
        num_samples = df.shape[0]
        num_features = df.shape[1] - 1  # 第一列是 ID
        summary_data.append((feature_type, num_samples, num_features))
        print(f"{feature_type:<10} {num_samples:<10} {num_features:<10}")
    else:
        print(f"[跳过] 文件未找到: {file}")

# 保存为 TXT
with open("features_summary.txt", "w") as f:
    f.write("特征类型\t样本数\t特征维度\n")
    for row in summary_data:
        f.write(f"{row[0]}\t{row[1]}\t{row[2]}\n")

# 保存为 CSV
summary_df = pd.DataFrame(summary_data, columns=["Feature", "Samples", "Dimensions"])
summary_df.to_csv("features_summary.csv", index=False)

print("\n📁 汇总已保存：features_summary.txt, features_summary.csv")
