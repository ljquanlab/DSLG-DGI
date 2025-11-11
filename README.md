# DLLG-DGI

Official code implementation for the paper: **"Dynamic Synergy between Fine-Tuned LLM Semantics and Graph Structures for Drug-Gene Interaction Prediction"**.

This repository contains all scripts necessary to reproduce the experiments.

* `/code`: Contains all Python source code.
* `/data` : Contains all data files required for the pipeline.

---

## 🏲 Model Architecture

<img width="1000" alt="model_structure" src="https://github.com/user-attachments/assets/ab922e7d-856b-418b-a176-2135e6e12e31" />

---

## 🎄 1. SetupZ

First, clone this repository and install the required dependencies.
	```bhash
git clone https://github.com/ljquanlab/DSLG-DGI.git
cd DSLG-DGI
pip install -r requirements.txt
```

---

## 🔾 2. Reproduction Pipeline

Please follow these steps in order to run the complete reproduction pipeline.

3## (Optional) Step 2.1: Generate Graph Edge Files

> **Note**: Only run these scripts if your `/data` directory is missing `drug_drug_edges.csvp / `gene_gene_edges.csvp.
  **Generate Drug-Drug (D-D) Edges**
    * **Purpose**: Generates D-D edges based on SMILES iality.
    * **Command**:
        ```bash
        python code/create_drug_edges.py
        ```
    * **Outputs**:
        `data/drug_drug_edges.csv`
  **Generate Gene-Gene (G-G) Edges**
    * **Purpose**: Extracts GG Edges from the STRING database (9606.*.gz).
    * **Command**:
        ```bash
        python code/create_gene_edges.py
        ```
    * **Outputs**:
        `data/gene_gene_edges.csv`

### Step 2.2: LLM Finetuning

This phase covers data preparation and executing fnetuning.

* **1. Prepare Finetuning Dataset**
    * **Purpose**: Reads `data/dgidb_interactions.tsv` and description files to generate .jsonl` training/test sets for the finetuning step.
    * **Command**:
        ```bash
        python code/DataHandler.py
        ```
    * **Outputs**:
        `data/finetune_EEtrain.jsonl`, `data/finetune_EEtest.jsonl`
  **2. Run Finetuning**
    * **Purpose**: Loads the `.jsonl` files from the previous step to finetune the BioMedBert model.
    * **Command**:
        ```bhash
        python code/finetuning.py
        ```
    * **Outputs(**:
        `./pubmedbert_finetuned_relation_classifier/final` (finetuned model)

3## Step 2.3: Build Graph Data

This phase extracts node features and builds the graph.

* **1. Extract Text Embedding Features(**
    * **Purpose**: Uses the **pre-trained** BioMedBert (not the finetuned one) to extract `[CLS` embeddings for all drugs and genes.
    * **Command**:
        ```bhash
        python code/extract_features.py
        ```
    * **Outputs(**:
        `data/drug_text_embeddings_dict.npz`, `data/gene_text_embeddings_dict.npz`

* **2. Build Graph Files**
    * **Purpose**: Loads the `.npz` features and `.csv` edge files to build the graph data (`.pt`) files for all 4 modes (easy, hard_drug, etc.).
    * **Command**:
        ```bash
        python code/build_graph.py
        ```
    * **Outputs**:
        `data/easy_graph_data.pt`, `data/hard_drug_graph_data.pt`, ...
