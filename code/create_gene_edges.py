import numpy as np
import pandas as pd
import os
import gzip

output_dir = '../data'
gene_embeddings_path = os.path.join(output_dir, "gene_text_embeddings_dict.npz")
aliases_path = os.path.join(output_dir, "9606.protein.aliases.v12.0.txt.gz")
links_path = os.path.join(output_dir, "9606.protein.links.v12.0.txt.gz")

confidence_threshold = 700

try:
    with np.load(gene_embeddings_path) as data:
        project_genes = set(data.keys())
except FileNotFoundError:
    exit()

string_id_to_gene_symbol = {}
try:
    with gzip.open(aliases_path, 'rt', encoding='utf-8') as f:
        next(f) 
        for line in f:
            string_id, alias, source = line.strip().split('\t')
            if alias in project_genes:
                string_id_to_gene_symbol[string_id] = alias
except FileNotFoundError:
    exit()

edge_list = []
try:
    with gzip.open(links_path, 'rt', encoding='utf-8') as f:
        next(f) 
        for line in f:
            protein1, protein2, combined_score = line.strip().split()
            score = int(combined_score)

            if score < confidence_threshold:
                continue

            gene1 = string_id_to_gene_symbol.get(protein1)
            gene2 = string_id_to_gene_symbol.get(protein2)

            if gene1 and gene2 and gene1 != gene2:
                edge_list.append((gene1, gene2, score))

except FileNotFoundError:
    exit()

output_path = os.path.join(output_dir, "gene_gene_edges.csv")
edge_df = pd.DataFrame(edge_list, columns=['gene_1', 'gene_2', 'score'])

edge_df['sorted_genes'] = edge_df.apply(lambda row: tuple(sorted((row['gene_1'], row['gene_2']))), axis=1)
edge_df.drop_duplicates(subset='sorted_genes', keep='first', inplace=True)
edge_df.drop(columns='sorted_genes', inplace=True)

edge_df.to_csv(output_path, index=False)