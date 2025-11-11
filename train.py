import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
from model import GatedFusionModel
from torch_geometric.data import Data

MODES_TO_RUN = [
    'easy',
    'hard_drug',
    'hard_gene',
    'hard_drug_gene'
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 1e-4
EPOCHS = 200
BATCH_SIZE = 4096
IN_CHANNELS = 768
GNN_HIDDEN_CHANNELS = 128 
GNN_OUT_CHANNELS = 768 
PREDICTOR_HIDDEN_CHANNELS = 256
GAT_HEADS = 4

output_dir = '../data'
drug_names = sorted(list(np.load(os.path.join(output_dir, "drug_text_embeddings_dict.npz")).keys()))
gene_names = sorted(list(np.load(os.path.join(output_dir, "gene_text_embeddings_dict.npz")).keys()))
all_node_names = drug_names + gene_names
node_to_idx = {name: i for i, name in enumerate(all_node_names)}

for mode in MODES_TO_RUN:
    print(f"\n\n{'='*20} Processing mode: '{mode}' {'='*20}")

    print(f"--- Loading data for '{mode}' mode ---")
    try:
        data = torch.load(os.path.join(output_dir, f'{mode}_graph_data.pt'), weights_only=False)
        train_df = pd.read_csv(os.path.join(output_dir, f"{mode}_train_df.csv"))
        val_df = pd.read_csv(os.path.join(output_dir, f"{mode}_val_df.csv"))
        test_df = pd.read_csv(os.path.join(output_dir, f"{mode}_test_df.csv"))
        print(f"Data for '{mode}' mode loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Data files for '{mode}' not found. Skipping. Please run build_graph.py first.")
        continue 

    data = data.to(DEVICE)

    model = GatedFusionModel(
        text_channels=IN_CHANNELS,
        gnn_hidden_channels=GNN_HIDDEN_CHANNELS,
        gnn_out_channels=GNN_OUT_CHANNELS,
        predictor_hidden_channels=PREDICTOR_HIDDEN_CHANNELS,
        heads=GAT_HEADS
    ).to(DEVICE)

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    criterion = torch.nn.BCEWithLogitsLoss()
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    print(f"Model and optimizer for '{mode}' initialized.")

    
    def get_edges_and_labels_from_df(df):
        pos_edges = df[df['label'] == 1]
        neg_edges = df[df['label'] == 0]
        pos_u = [node_to_idx.get(name) for name in pos_edges['drug_name']]
        pos_v = [node_to_idx.get(name) for name in pos_edges['gene_name']]
        neg_u = [node_to_idx.get(name) for name in neg_edges['drug_name']]
        neg_v = [node_to_idx.get(name) for name in neg_edges['gene_name']]
        
        pos_u_v = [(u, v) for u, v in zip(pos_u, pos_v) if u is not None and v is not None]
        neg_u_v = [(u, v) for u, v in zip(neg_u, neg_v) if u is not None and v is not None]
        
        pos_u, pos_v = zip(*pos_u_v) if pos_u_v else ([], [])
        neg_u, neg_v = zip(*neg_u_v) if neg_u_v else ([], [])

        edge_u = torch.tensor(list(pos_u) + list(neg_u), dtype=torch.long)
        edge_v = torch.tensor(list(pos_v) + list(neg_v), dtype=torch.long)
        labels = torch.tensor([1] * len(pos_u) + [0] * len(neg_u), dtype=torch.float)
        return edge_u, edge_v, labels

    def train():
        model.train()
        train_u, train_v, train_labels = get_edges_and_labels_from_df(train_df)
        train_labels = train_labels.to(DEVICE)
        perm = torch.randperm(len(train_u))
        train_u, train_v, train_labels = train_u[perm], train_v[perm], train_labels[perm]

        total_loss = 0
        pbar = tqdm(range(0, len(train_u), BATCH_SIZE), desc=f"Training ({mode})")
        for i in pbar:
            optimizer.zero_grad()
            batch_u, batch_v = train_u[i:i+BATCH_SIZE].to(DEVICE), train_v[i:i+BATCH_SIZE].to(DEVICE)
            batch_labels = train_labels[i:i+BATCH_SIZE].to(DEVICE)
            out = model(data.x, data.edge_index, batch_u, batch_v)
            loss = criterion(out, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(batch_u)
            pbar.set_postfix({'loss': loss.item()})
        return total_loss / len(train_u)

    @torch.no_grad()
    def test(df_to_test):
        model.eval()
        test_u, test_v, test_labels = get_edges_and_labels_from_df(df_to_test)
        
        if len(test_u) == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
        test_labels_np = test_labels.cpu().numpy()
        
        preds_logits = []
        pbar = tqdm(range(0, len(test_u), BATCH_SIZE), desc=f"Evaluating ({mode})")
        for i in pbar:
            batch_u, batch_v = test_u[i:i+BATCH_SIZE].to(DEVICE), test_v[i:i+BATCH_SIZE].to(DEVICE)
            out = model(data.x, data.edge_index, batch_u, batch_v)
            preds_logits.append(out.cpu())
        preds_logits = torch.cat(preds_logits, dim=0)
        
        loss = criterion(preds_logits, test_labels)
        preds_probs = preds_logits.sigmoid().numpy()
        preds_binary = (preds_probs > 0.5).astype(int)
        
        precision, recall, f1, _ = precision_recall_fscore_support(test_labels_np, preds_binary, average='binary', zero_division=0)
        
        auc = roc_auc_score(test_labels_np, preds_probs)
        ap = average_precision_score(test_labels_np, preds_probs)
        
        return loss.item(), auc, ap, precision, recall, f1

    print(f"--- Starting final model training for '{mode}' mode ---")
    best_val_auc = 0
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss = train()
        val_loss, val_auc, val_ap, val_prec, val_rec, val_f1 = test(val_df)
        scheduler.step(val_loss)
        
        print(f"Epoch: {epoch:03d}, Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Val AUC: {val_auc:.4f}, Val AUPR: {val_ap:.4f}, Val F1: {val_f1:.4f}, "
              f"Val Prec: {val_prec:.4f}, Val Recall: {val_rec:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            torch.save(model.state_dict(), f'best_model_{mode}_final.pt')
            print(f"  (New best! Saved model for '{mode}' at epoch {epoch})")

    print(f"\nTraining for '{mode}' complete. Best Val AUC: {best_val_auc:.4f} (at epoch {best_epoch})")

    print(f"\nEvaluating best model for '{mode}' on test set...")
    model.load_state_dict(torch.load(f'best_model_{mode}_final.pt'))
    _, test_auc, test_ap, test_prec, test_rec, test_f1 = test(test_df)
    print(f"--- Final Test Set Results for '{mode}' ---")
    print(f"AUC: {test_auc:.4f}, AUPR: {test_ap:.4f}, F1: {test_f1:.4f}, "
          f"Precision: {test_prec:.4f}, Recall: {test_rec:.4f}")

print("\n\nAll specified modes have been trained and evaluated!")