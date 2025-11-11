# DSLG-DGI

Official code implementation for the paper: "Dynamic Synergy between Fine-Tuned LLM Semantics and Graph Structures for Drug–Gene Interaction Prediction".

This repository contains all scripts necessary to reproduce the experiments.
- `/code`: Contains all Python source code.
- `/data`: Contains all data files required for the pipeline.

---

## 1. Setup

Clone this repository and install the required dependencies from `requirements.txt`.
<img width="2496" height="2992" alt="model_structure" src="https://github.com/user-attachments/assets/ab922e7d-856b-418b-a176-2135e6e12e31" />

# ----------------------------------------------------------------------
# (可选) 如果 data/ 目录中没有 drug_drug_edges.csv
# 作用: 基于SMILES相似性，生成药物-药物(D-D)边
# 生成: data/drug_drug_edges.csv
# ----------------------------------------------------------------------
python code/create_drug_edges.py

# ----------------------------------------------------------------------
# (可选) 如果 data/ 目录中没有 gene_gene_edges.csv
# 作用: 从STRING数据库(9606.*.gz)中提取，生成基因-基因(G-G)边
# 生成: data/gene_gene_edges.csv
# ----------------------------------------------------------------------
python code/create_gene_edges.py

# ----------------------------------------------------------------------
# 作用: 读取 data/dgidb_interactions.tsv 和描述文件，
#       为微调(finetuning)步骤生成 .jsonl 训练/测试集
# 生成: data/finetune_EEtrain.jsonl, data/finetune_EEtest.jsonl
# ----------------------------------------------------------------------
python code/DataHandler.py
# ----------------------------------------------------------------------
# 作用: 加载第1步生成的 .jsonl 文件，微调 BioMedBert 模型
# 生成: ./pubmedbert_finetuned_relation_classifier/final (微调好的模型)
# ----------------------------------------------------------------------
python code/finetuning.py
# ----------------------------------------------------------------------
# 作用: 使用预训练的 BioMedBert 为所有药物和基因提取 [CLS] 嵌入
#       (注意: 此脚本默认使用 *预训练* 模型，而非 *微调* 模型)
# 生成: data/drug_text_embeddings_dict.npz
#       data/gene_text_embeddings_dict.npz
# ----------------------------------------------------------------------
python code/extract_features.py
# ----------------------------------------------------------------------
# 作用: 加载第3步的 .npz 特征 和 第1步的 .csv 边文件，
#       为所有4种模式 (easy, hard_drug 等) 构建图数据 (.pt) 文件
# 生成: data/easy_graph_data.pt, data/hard_drug_graph_data.pt, ...
# ----------------------------------------------------------------------
python code/build_graph.py

# ----------------------------------------------------------------------
# 作用: 加载 .pt 图文件和交互数据 (train_df.csv 等)，
#       为所有4种模式运行最终的 DSLG-DGI 模型训练和评估
# 生成: best_model_easy_final.pt, ...
# ----------------------------------------------------------------------
python code/train.py


```bash
git clone [https://github.com/ljquanlab/DSLG-DGI.git](https://github.com/ljquanlab/DSLG-DGI.git)
cd DSLG-DGI
pip install -r requirements.txt

