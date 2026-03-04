import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool

class RankGNN(torch.nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, output_dim=10, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.convs = torch.nn.ModuleList([
            GCNConv(input_dim, hidden_dim),
            GCNConv(hidden_dim, hidden_dim),
            GCNConv(hidden_dim, hidden_dim),
            GCNConv(hidden_dim, hidden_dim)
        ])
        self.pool = global_mean_pool
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, output_dim)
        )
        self.aux_head = torch.nn.Linear(hidden_dim, 1)
    
    def forward(self, x, edge_index, batch, edge_weight=None):
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight)
            x = torch.relu(x)
            x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        graph_emb = self.pool(x, batch)
        rank_logits = self.classifier(graph_emb)
        stability_score = torch.sigmoid(self.aux_head(graph_emb)).squeeze(1)
        return rank_logits, stability_score

class SimpleGNN(torch.nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=10):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.pool = global_mean_pool
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, edge_index, batch, edge_weight=None):
        x = torch.relu(self.conv1(x, edge_index, edge_weight))
        x = torch.relu(self.conv2(x, edge_index, edge_weight))
        x = self.pool(x, batch)
        x = self.fc(x)
        return x, None