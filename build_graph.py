import torch as t
import numpy as np
import pandas as pd
from torch_geometric.data import Data
import os
from DataHandler import DataHandler 

MODES_TO_RUN = [
    'easy',   
    'hard_drug',
    'hard_gene',
    'hard_drug_gene'
]

output_dir = "../data"
drug_gene_interaction_file = os.path.join(output_dir, 'dgidb_interactions.tsv')

with np.load(os.path.join(output_dir, "drug_text_embeddings_dict.npz")) as loader:
    drug_embeddings = dict(loader)
with np.load(os.path.join(output_dir, "gene_text_embeddings_dict.npz")) as loader:
    gene_embeddings = dict(loader)
drug_drug_edges = pd.read_csv(os.path.join(output_dir, "drug_drug_edges.csv"))
gene_gene_edges = pd.read_csv(os.path.join(output_dir, "gene_gene_edges.csv"))

drug_names = sorted(list(drug_embeddings.keys()))
gene_names = sorted(list(gene_embeddings.keys()))
all_node_names = drug_names + gene_names
node_to_idx = {name: i for i, name in enumerate(all_node_names)}
num_nodes = len(all_node_names)
feature_dim = list(drug_embeddings.values())[0].shape[0]

x = t.zeros((num_nodes, feature_dim), dtype=t.float)
for name, idx in node_to_idx.items():
    if name in drug_embeddings:
        x[idx] = t.from_numpy(drug_embeddings[name])
    elif name in gene_embeddings:
        x[idx] = t.from_numpy(gene_embeddings[name])

for mode in MODES_TO_RUN:
    
    data_handler = DataHandler(
        file_path=drug_gene_interaction_file,
        mode=mode, 
        val_size=0.1,
        test_size=0.1
    )
    train_df, val_df, test_df = data_handler.load_data()
    
    file_prefix = mode
    train_df.to_csv(os.path.join(output_dir, f"{file_prefix}_train_df.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, f"{file_prefix}_val_df.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, f"{file_prefix}_test_df.csv"), index=False)
    
    edge_list = []
    
    if 'hard' in mode:
        train_drug_nodes = set(train_df['drug_name'].unique())
        train_gene_nodes = set(train_df['gene_name'].unique())
        train_nodes_set = train_drug_nodes | train_gene_nodes
        
        dd_edges_to_add = drug_drug_edges[drug_drug_edges['drug_1'].isin(train_nodes_set) & drug_drug_edges['drug_2'].isin(train_nodes_set)]
        gg_edges_to_add = gene_gene_edges[gene_gene_edges['gene_1'].isin(train_nodes_set) & gene_gene_edges['gene_2'].isin(train_nodes_set)]
    else: 
        dd_edges_to_add = drug_drug_edges
        gg_edges_to_add = gene_gene_edges

    for _, row in dd_edges_to_add.iterrows():
        u, v = node_to_idx.get(row['drug_1']), node_to_idx.get(row['drug_2'])
        if u is not None and v is not None:
            edge_list.append([u, v]); edge_list.append([v, u])
    
    for _, row in gg_edges_to_add.iterrows():
        u, v = node_to_idx.get(row['gene_1']), node_to_idx.get(row['gene_2'])
        if u is not None and v is not None:
            edge_list.append([u, v]); edge_list.append([v, u])

    positive_train_edges = train_df[train_df['label'] == 1]
    for _, row in positive_train_edges.iterrows():
        u, v = node_to_idx.get(row['drug_name']), node_to_idx.get(row['gene_name'])
        if u is not None and v is not None:
            edge_list.append([u, v]); edge_list.append([v, u])

    edge_index = t.tensor(edge_list, dtype=t.long).t().contiguous()
    
    graph_data = Data(x=x, edge_index=edge_index)
    graph_data_path = os.path.join(output_dir, f'{file_prefix}_graph_data.pt')
    t.save(graph_data, graph_data_path)