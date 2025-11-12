# OGTPred-ESM
OGT预测模型是一个基于深度学习的蛋白质最适生长温度预测工具。本项目创新性地融合了ESM-2蛋白质语言模型的深度序列表征与传统的理化特征，通过特征融合策略构建了一个高精度的OGT预测模型。

## esm2对数据集进行序列表征的脚本
./embeding。py

## 进行基准机器学习训练与测试的脚本
./ogt_ml_base.ipynb

## 最终模型训练文件
OGTPred-ESM/script/02_pca_ifeature.py


## 最终模型 02_pca_ifeature 测试结果文件路径
OGTPred-ESM/result/02_ogt_esm_pca_physico
