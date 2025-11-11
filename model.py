import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GatedFusionModel(nn.Module):
    def __init__(self, text_channels, gnn_hidden_channels, gnn_out_channels, predictor_hidden_channels, heads):
        super(GatedFusionModel, self).__init__()

        self.gnn_conv1 = GATConv(text_channels, gnn_hidden_channels, heads=heads)
        self.gnn_conv2 = GATConv(gnn_hidden_channels * heads, gnn_out_channels, heads=1)

        gate_input_dim = text_channels + gnn_out_channels
        self.gate_nn = nn.Sequential(
            nn.Linear(gate_input_dim, gate_input_dim // 2),
            nn.ReLU(),
            nn.Linear(gate_input_dim // 2, 1),
            nn.Sigmoid()
        )
        
        predictor_input_dim = gnn_out_channels * 2
        self.predictor = nn.Sequential(
            nn.Linear(predictor_input_dim, predictor_hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(predictor_hidden_channels, 1)
        )

    def forward(self, x, edge_index, u_nodes, v_nodes):
        
        h_graph = self.gnn_conv1(x, edge_index)
        h_graph = F.elu(h_graph)
        h_graph = self.gnn_conv2(h_graph, edge_index)

        gate_input = torch.cat([x, h_graph], dim=-1)
        g = self.gate_nn(gate_input)

        h_final = (1 - g) * x + g * h_graph

        u_embeds = h_final[u_nodes]
        v_embeds = h_final[v_nodes]
        
        pair_embeds = torch.cat([u_embeds, v_embeds], dim=-1)
        
        out = self.predictor(pair_embeds)
        
        return out.squeeze(-1)