import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphTransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super(GraphTransformerLayer, self).__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, adj):
        # x: [batch, num_nodes, hidden_dim]
        # adj: [num_nodes, num_nodes] (adjacency matrix)
        batch_size, num_nodes, hidden_dim = x.shape
        
        # Apply attention, masking non-neighbors
        adj_mask = (adj == 0).unsqueeze(0).expand(batch_size, -1, -1)  # [batch, num_nodes, num_nodes]
        x_residual = x
        x, _ = self.attention(x, x, x, attn_mask=adj_mask)
        x = self.norm1(x_residual + self.dropout(x))
        
        # Feedforward
        x_residual = x
        x = self.ffn(x)
        x = self.norm2(x_residual + self.dropout(x))
        return x

class GraphTransformerClassifier(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_classes=4, num_heads=4, num_layers=2, dropout=0.2):
        super(GraphTransformerClassifier, self).__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([
            GraphTransformerLayer(hidden_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        self.fc1 = nn.Linear(hidden_dim * 21, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        # Define adjacency matrix for hand graph
        self.adj = self.create_hand_adjacency_matrix()
    
    def create_hand_adjacency_matrix(self):
        adj = torch.zeros(21, 21)
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (5, 9), (9, 13), (13, 17)  # Palm connections
        ]
        for i, j in connections:
            adj[i, j] = 1
            adj[j, i] = 1
        # Add self-loops
        adj += torch.eye(21)
        return adj
    
    def forward(self, x):
        # x: [batch, 21, 2]
        adj = self.adj.to(x.device)
        x = self.relu(self.input_proj(x))  # [batch, 21, hidden_dim]
        
        # Apply graph transformer layers
        for layer in self.layers:
            x = layer(x, adj)
        
        # Pool and classify
        x = x.view(x.size(0), -1)  # [batch, 21 * hidden_dim]
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x