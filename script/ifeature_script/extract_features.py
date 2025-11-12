import os
import subprocess
import pandas as pd

# 输入/输出路径
input_fasta = '/mnt/data/lzp/OGT/data/brenda/protein_sequences.fasta'
output_file = '/mnt/data/lzp/OGT/result/ifeature_result/protein_features.csv'
ifeature_script = '/mnt/data/lzp/software/iFeature/iFeature.py'

# 原始特征类型列表
raw_feature_types = [
    'AAC',
    'CKSAAP',
    'DPC',
    'PSSM',
    'CTD',  # 自动展开为 CTDC, CTDT, CTDD
    'QSOrder',
    'PAAC',
    'Kmer',
    'AACI',
]

# 将 CTD 拆分为三个子类型
feature_types = []
for ft in raw_feature_types:
    if ft == 'CTD':
        feature_types.extend(['CTDC', 'CTDT', 'CTDD'])
    else:
        feature_types.append(ft)

# 用于保存所有特征 DataFrame（自动对齐 ID）
all_features = []
id_column = 'SampleName'

# 主循环：逐个特征提取
for feature_type in feature_types:
    out_file = f'{feature_type}_features.tsv'
    command = [
        'python3', ifeature_script,
        '--file', input_fasta,
        '--type', feature_type,
        '--out', out_file
    ]

    print(f'\n▶ 正在提取特征: {feature_type} ...')
    try:
        subprocess.run(command, check=True)

        # 检查输出文件是否存在且非空
        if not os.path.exists(out_file) or os.path.getsize(out_file) == 0:
            print(f'[警告] 特征 {feature_type} 输出文件为空，跳过。')
            continue

        # 读取并合并
        df = pd.read_csv(out_file, sep='\t')
        first_col = df.columns[0]  # 自动识别第一列
        df = df.set_index(first_col)
        all_features.append(df)
        print(f'✅ 特征提取成功: {feature_type}')




        
        
    except subprocess.CalledProcessError:
        print(f'[错误] 特征提取失败（跳过）: {feature_type}')

# 合并所有特征
if all_features:
    combined_df = pd.concat(all_features, axis=1)
    combined_df.reset_index(inplace=True)
    combined_df.to_csv(output_file, index=False)
    print(f'\n🎉 所有有效特征已保存到: {output_file}')
else:
    print('\n❌ 没有成功提取任何特征，未生成输出文件。')
