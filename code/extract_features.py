import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext" 

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

def build_name_to_desc_map(names_filepath, descs_filepath):
    name_to_desc = {}
    try:
        with open(names_filepath, 'r', encoding='utf-8') as f_names, \
             open(descs_filepath, 'r', encoding='utf-8') as f_descs:
            
            names_list = [line.strip() for line in f_names]
            descs_lines_list = [line.strip() for line in f_descs]

            if len(names_list) != len(descs_lines_list):
                exit()
            
            for i in range(len(names_list)):
                correct_name = names_list[i]
                full_desc_line = descs_lines_list[i]
                
                if ':' in full_desc_line:
                    _, description = full_desc_line.rsplit(':', 1)
                    name_to_desc[correct_name] = description.strip()
                else:
                    name_to_desc[correct_name] = ""
            
    except FileNotFoundError as e:
        exit()

    return name_to_desc

drug_name_desc_map = build_name_to_desc_map('../data/unique_drug_names_for_model.txt', '../data/drugwithEnglish.txt')
gene_name_desc_map = build_name_to_desc_map('../data/unique_gene_names_for_model.txt', '../data/genewithEnglish.txt')

batch_size = 32
hidden_size = model.config.hidden_size

def generate_embeddings_from_map(name_desc_map, pbar_desc):
    embeddings_dict = {}
    names_list = list(name_desc_map.keys())

    with torch.no_grad():
        for i in tqdm(range(0, len(names_list), batch_size), desc=pbar_desc):
            batch_names = names_list[i:i + batch_size]
            batch_texts = [name_desc_map.get(name, "") for name in batch_names] 

            inputs = tokenizer(
                batch_texts, padding=True, truncation=True, 
                max_length=512, return_tensors="pt"
            ).to(device)

            outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]

            for idx, name in enumerate(batch_names):
                if not name_desc_map.get(name):
                    embeddings_dict[name] = np.zeros(hidden_size, dtype=np.float32)
                else:
                    embeddings_dict[name] = cls_embeddings[idx].cpu().numpy()
    return embeddings_dict

drug_embeddings = generate_embeddings_from_map(drug_name_desc_map, "Processing Drugs")
gene_embeddings = generate_embeddings_from_map(gene_name_desc_map, "Processing Genes")

output_dir = '../data'
os.makedirs(output_dir, exist_ok=True)
drug_output_path = os.path.join(output_dir, "xiao_drug_text_embeddings_dict.npz")
gene_output_path = os.path.join(output_dir, "xiao_gene_text_embeddings_dict.npz")

try:
    np.savez_compressed(drug_output_path, **drug_embeddings)
    np.savez_compressed(gene_output_path, **gene_embeddings)
except Exception as e:
    pass