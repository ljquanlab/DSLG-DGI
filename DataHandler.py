import pandas as pd
from sklearn.model_selection import train_test_split
import json

class DataHandler:
    def __init__(self, file_path, mode='easy', val_size=0.1, test_size=0.1):
        self.file_path = file_path
        self.test_size = test_size
        self.val_size = val_size
        self.mode = mode
        self.random_state = 42

        self.data = None
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def load_data(self):
        self.data = pd.read_csv(self.file_path, sep='\t', encoding='utf-8')
        self.data['interaction_type'] = self.data['interaction_type'].astype(str)

        if self.mode == 'easy':
            train_val_data, self.test_data = train_test_split(
                self.data,
                test_size=self.test_size,
                random_state=self.random_state
            )
            val_proportion_in_rest = self.val_size / (1 - self.test_size)
            self.train_data, self.val_data = train_test_split(
                train_val_data,
                test_size=val_proportion_in_rest,
                random_state=self.random_state
            )
        
        elif self.mode == 'hard_drug':
            all_drugs = self.data['drug_name'].unique()
            
            train_drugs, temp_drugs = train_test_split(all_drugs, test_size=(self.val_size + self.test_size), random_state=self.random_state)
            val_proportion = self.val_size / (self.val_size + self.test_size)
            val_drugs, test_drugs = train_test_split(temp_drugs, test_size=(1 - val_proportion), random_state=self.random_state)
            
            self.train_data = self.data[self.data['drug_name'].isin(train_drugs)].copy()
            self.val_data = self.data[self.data['drug_name'].isin(val_drugs)].copy()
            self.test_data = self.data[self.data['drug_name'].isin(test_drugs)].copy()

        elif self.mode == 'hard_gene':
            all_genes = self.data['gene_name'].unique()

            train_genes, temp_genes = train_test_split(all_genes, test_size=(self.val_size + self.test_size), random_state=self.random_state)
            val_proportion = self.val_size / (self.val_size + self.test_size)
            val_genes, test_genes = train_test_split(temp_genes, test_size=(1 - val_proportion), random_state=self.random_state)

            self.train_data = self.data[self.data['gene_name'].isin(train_genes)].copy()
            self.val_data = self.data[self.data['gene_name'].isin(val_genes)].copy()
            self.test_data = self.data[self.data['gene_name'].isin(test_genes)].copy()

        elif self.mode == 'hard_drug_gene':
            all_drugs = self.data['drug_name'].unique()
            all_genes = self.data['gene_name'].unique()

            train_drugs, temp_drugs = train_test_split(all_drugs, test_size=(self.val_size + self.test_size), random_state=self.random_state)
            val_drug_prop = self.val_size / (self.val_size + self.test_size)
            val_drugs, test_drugs = train_test_split(temp_drugs, test_size=(1 - val_drug_prop), random_state=self.random_state)

            train_genes, temp_genes = train_test_split(all_genes, test_size=(self.val_size + self.test_size), random_state=self.random_state)
            val_gene_prop = self.val_size / (self.val_size + self.test_size)
            val_genes, test_genes = train_test_split(temp_genes, test_size=(1 - val_gene_prop), random_state=self.random_state)
            
            self.train_data = self.data[self.data['drug_name'].isin(train_drugs) & self.data['gene_name'].isin(train_genes)].copy()
            self.val_data = self.data[self.data['drug_name'].isin(val_drugs) & self.data['gene_name'].isin(val_genes)].copy()
            self.test_data = self.data[self.data['drug_name'].isin(test_drugs) & self.data['gene_name'].isin(test_genes)].copy()
        
        else:
            raise ValueError(f"Unknown mode: '{self.mode}'.")

        self.train_data = self.train_data.reset_index(drop=True)
        self.val_data = self.val_data.reset_index(drop=True)
        self.test_data = self.test_data.reset_index(drop=True)
        
        if 'hard' in self.mode:
            self._verify_split()

        return self.train_data, self.val_data, self.test_data
    
    def _verify_split(self):
        train_drugs_set = set(self.train_data['drug_name'].unique())
        val_drugs_set = set(self.val_data['drug_name'].unique())
        test_drugs_set = set(self.test_data['drug_name'].unique())
        
        train_genes_set = set(self.train_data['gene_name'].unique())
        val_genes_set = set(self.val_data['gene_name'].unique())
        test_genes_set = set(self.test_data['gene_name'].unique())

        if self.mode == 'hard_drug':
            assert len(train_drugs_set.intersection(val_drugs_set)) == 0
            assert len(train_drugs_set.intersection(test_drugs_set)) == 0
        elif self.mode == 'hard_gene':
            assert len(train_genes_set.intersection(val_genes_set)) == 0
            assert len(train_genes_set.intersection(test_genes_set)) == 0
        elif self.mode == 'hard_drug_gene':
            assert len(train_drugs_set.intersection(val_drugs_set)) == 0
            assert len(train_drugs_set.intersection(test_drugs_set)) == 0
            assert len(train_genes_set.intersection(val_genes_set)) == 0
            assert len(train_genes_set.intersection(test_genes_set)) == 0


class LLMDataHandler():
    def __init__(self, drug_desc_path, gene_desc_path):
        self.drug_desc_path = drug_desc_path 
        self.gene_desc_path = gene_desc_path 

        self.drug_desc_dict = {}
        self.gene_desc_dict = {}

    def _read_descriptions_to_dict(self, filepath, encoding):
        descriptions_dict = {}
        with open(filepath, 'r', encoding=encoding) as f:
            for line in f:
                stripped_line = line.strip()
                if stripped_line:
                    split_parts = stripped_line.rsplit(':', 1) 
                    if len(split_parts) == 2:
                        name = split_parts[0].strip()
                        description = split_parts[1].strip()
                        descriptions_dict[name] = description
        return descriptions_dict

    def _save_to_jsonl(self, data, filename):
        with open(filename, 'w', encoding='utf-8') as f: 
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    def _generate_jsonl_from_df(self, dataframe_to_process): 
        jsonl_output_list = []
        skipped_count = 0

        for _, row in dataframe_to_process.iterrows():
            drug_name = row['drug_name']
            gene_name = row['gene_name']
            
            drug_desc = self.drug_desc_dict.get(drug_name)
            gene_desc = self.gene_desc_dict.get(gene_name)

            if not drug_desc or not gene_desc:
                skipped_count += 1
                continue

            input_text = f"[CLS] {drug_desc} [SEP] {gene_desc} [SEP]"
            label = int(row['label'])
            jsonl_output_list.append({'text': input_text, 'label': label})
        
        return jsonl_output_list

    def process_data(self, train_df, test_df, train_output_path, test_output_path):
        self.drug_desc_dict = self._read_descriptions_to_dict(self.drug_desc_path, 'utf-8')
        self.gene_desc_dict = self._read_descriptions_to_dict(self.gene_desc_path, 'utf-8')
        if not self.drug_desc_dict or not self.gene_desc_dict:
            return

        finetuning_train_data = self._generate_jsonl_from_df(train_df)
        
        finetuning_test_data = self._generate_jsonl_from_df(test_df)
        
        if finetuning_train_data:
            self._save_to_jsonl(finetuning_train_data, train_output_path)
        if finetuning_test_data:
            self._save_to_jsonl(finetuning_test_data, test_output_path)

if __name__ == "__main__":
    test_mode = 'easy'

    dh = DataHandler(file_path="../data/dgidb_interactions.tsv", mode=test_mode)
    train_df, val_df, test_df = dh.load_data()

    llm_dh = LLMDataHandler(
        drug_desc_path="../data/drugwithEnglish.txt",
        gene_desc_path="../data/genewithEnglish.txt"
    )

    llm_dh.process_data(
        train_df=train_df, 
        test_df=test_df,  
        train_output_path="../data/finetune_EEtrain.jsonl", 
        test_output_path="../data/finetune_EEtest.jsonl", 
    )