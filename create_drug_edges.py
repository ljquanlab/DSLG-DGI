import pickle
import pandas as pd
from rdkit import Chem
from rdkit.DataStructs import BulkTanimotoSimilarity
from rdkit.Chem import AllChem
import os

file_path = "../data/drug_smiles_map.pkl"
with open(file_path, 'rb') as f:
    drug_smiles_map = pickle.load(f)

smiles_df = pd.DataFrame(list(drug_smiles_map.items()), columns=['drug_name', 'smiles'])

smiles_df.dropna(subset=['smiles'], inplace=True) 
smiles_df = smiles_df[smiles_df['smiles'].apply(lambda x: isinstance(x, str) and x.strip() != '')]

mols = [Chem.MolFromSmiles(s) for s in smiles_df['smiles']]
valid_indices = [i for i, mol in enumerate(mols) if mol is not None]
mols = [mols[i] for i in valid_indices]
smiles_df_valid = smiles_df.iloc[valid_indices].reset_index(drop=True)

fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024) for m in mols]

similarity_threshold = 0.7
edge_list = []

for i in range(len(fps)):
    similarities = BulkTanimotoSimilarity(fps[i], fps[i+1:])
    
    for j, sim in enumerate(similarities):
        if sim >= similarity_threshold:
            original_j_index = i + 1 + j
            drug1_name = smiles_df_valid.loc[i, 'drug_name']
            drug2_name = smiles_df_valid.loc[original_j_index, 'drug_name']
            edge_list.append((drug1_name, drug2_name, round(sim, 4)))

output_dir = '../data'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "drug_drug_edges.csv")

edge_df = pd.DataFrame(edge_list, columns=['drug_1', 'drug_2', 'similarity'])
edge_df.to_csv(output_path, index=False)