import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_add_pool, global_mean_pool


'''
GAT- uses Attention strategy
compute the hidden representations of each node in the Graph by attending 
over its neighbors using a self-attention strategy
'''
class GAT(torch.nn.Module):
    def __init__(self, num_node_features, hidden_classes, num_classes, dropout=0.3):
        super(GAT, self).__init__()
        self.hid = hidden_classes
        self.first_heads = 5
        self.out_head = 3
        self.dropout = dropout
        
        self.conv1 = GATConv(num_node_features, self.hid, heads=self.first_heads, dropout=self.dropout)
        self.conv2 = GATConv(self.hid*self.first_heads, self.hid, heads=self.first_heads, dropout=self.dropout)
        self.conv3 = GATConv(self.hid*self.first_heads, self.hid, heads=self.out_head, concat=False, dropout=self.dropout)
        self.classifier = torch.nn.Linear(self.hid, num_classes)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # Dropout before the GAT layer is used to avoid overfitting
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        h = self.conv2(x, edge_index)
        x = F.elu(x+h)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, data.batch)
        x = self.classifier(x)
        return x,F.log_softmax(x, dim=1)

class EdgeGAT(torch.nn.Module):
    def __init__(self, num_node_features, hidden_classes, num_classes, edge_dim, dropout=0.3):
        super(EdgeGAT, self).__init__()
        self.hid = hidden_classes
        self.first_heads = 5
        self.out_head = 3
        self.dropout = dropout
        self.edge_dim = edge_dim
        self.conv1 = GATConv(num_node_features, self.hid, heads=self.first_heads, dropout=self.dropout, edge_dim=self.edge_dim)
        self.conv2 = GATConv(self.hid*self.first_heads, self.hid, heads=self.first_heads, dropout=self.dropout, edge_dim=self.edge_dim)
        self.conv3 = GATConv(self.hid*self.first_heads, self.hid, heads=self.out_head, concat=False, dropout=self.dropout, edge_dim=self.edge_dim)
        self.classifier = torch.nn.Linear(self.hid, num_classes)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        # Dropout before the GAT layer is used to avoid overfitting
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index, edge_weight)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        h = self.conv2(x, edge_index, edge_weight)
        x = F.elu(x+h)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index, edge_weight)
        x = global_mean_pool(x, data.batch)
        x = self.classifier(x)
        return x,F.log_softmax(x, dim=1)