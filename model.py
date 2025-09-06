
# -*- coding: utf-8 -*-
# model.py — Define GNN models for multi-modal skin lesion classification
# type: ignore

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphSAGE, GATConv, GINConv


class GNNClassifier(torch.nn.Module):
    """
    Default GCN-based classifier (2 layers + FC).
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.4):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
    
# Placeholder for GNN with Gating Fusion
class GatingFusionGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.4):
        super(GatingFusionGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = dropout
        # TODO: Add gating fusion logic here

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # TODO: Apply gating fusion to x before GNN layers
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

# Placeholder for GNN with Cross-Attention Fusion
class CrossAttentionFusionGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.4):
        super(CrossAttentionFusionGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = dropout
        # TODO: Add cross-attention fusion logic here

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # TODO: Apply cross-attention fusion to x before GNN layers
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

# Placeholder for GNN with Adaptive Fusion
class AdaptiveFusionGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.4, split_indices=None):
        super(AdaptiveFusionGNN, self).__init__()
        self.dropout = dropout
        self.split_indices = split_indices
        if self.split_indices:
            self.num_modalities = len(self.split_indices)
            self.proj_layers = torch.nn.ModuleList([
                torch.nn.Linear(end - start, hidden_dim) for (start, end) in self.split_indices
            ])
        else:
            self.num_modalities = 1
            self.proj_layers = torch.nn.ModuleList([torch.nn.Linear(input_dim, hidden_dim)])
        self.alpha = torch.nn.Parameter(torch.ones(self.num_modalities))
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # Tách x thành các loại đặc trưng theo split_indices
        if self.split_indices:
            x_modalities = [x[:, start:end] for (start, end) in self.split_indices]
            x_proj = [proj(xm) for proj, xm in zip(self.proj_layers, x_modalities)]
            alpha_weights = torch.softmax(self.alpha, dim=0)
            # Hỗ trợ số lượng modalities tuỳ ý
            x_fused = sum(alpha_weights[i] * x_proj[i] for i in range(self.num_modalities))
        else:
            x_fused = self.proj_layers[0](x)
        x = F.relu(self.bn1(self.conv1(x_fused, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

# Placeholder for GNN with Local Graph Convolution (LGC)
class LGC_GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.4):
        super(LGC_GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = dropout
        # TODO: Add LGC logic here

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # TODO: Apply LGC logic to x before/after GNN layers
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)



class GFDHCM_GNN(GNNClassifier):
    """
    GCN (2 layers + FC, hidden_dim=64, dropout=0.4) using deep, handcrafted, and clinical features. Features are fused by concatenation. No adaptive fusion, no LGC.
    """
    def __init__(self, input_dims, hidden_dim, num_classes, dropout):
        super().__init__(input_dims, hidden_dim, num_classes, dropout)

class GFC_GNN(GNNClassifier):
    """
    GCN (2 layers + FC, hidden_dim=64, dropout=0.4) using only clinical features (age, sex, localization). No deep or handcrafted features. No fusion, no LGC.
    """
    def __init__(self, input_dims, hidden_dim, num_classes, dropout):
        super().__init__(input_dims, hidden_dim, num_classes, dropout)

class GFC_GNN(GNNClassifier):
    # Duplicate class removed

    # ...existing code...
    """
    Default GCN-based classifier (2 layers + FC).
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.4):
        super(GNNClassifier, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class GraphSAGEClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.4):
        super(GraphSAGEClassifier, self).__init__()
        from torch_geometric.nn import SAGEConv
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class GATClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=4, dropout=0.4):
        super(GATClassifier, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim * heads)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.elu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.bn2(self.conv2(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class GINClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.4):
        super(GINClassifier, self).__init__()
        nn1 = torch.nn.Sequential(torch.nn.Linear(input_dim, hidden_dim),
                                  torch.nn.ReLU(),
                                  torch.nn.Linear(hidden_dim, hidden_dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)

        nn2 = torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim),
                                  torch.nn.ReLU(),
                                  torch.nn.Linear(hidden_dim, hidden_dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)

        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
