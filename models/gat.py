import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_add_pool


'''
GAT- uses Attention stratgey
compute the hidden representations of each node in the Graph by attending 
over its neighbors using a self-attention strategy
'''
class GAT(torch.nn.Module):
    def __init__(self, num_node_features, hidden_classes, num_classes, dropout=0.3):
        super(GAT, self).__init__()
        self.hid = 300
        self.first_heads = 5
        self.out_head = 3
        self.dropout = dropout
        
        self.conv1 = GATConv(num_node_features, self.hid, heads=self.first_heads, dropout=self.dropout)
        self.conv2 = GATConv(self.hid*self.first_heads, self.hid, heads=self.first_heads, dropout=self.dropout)
        self.conv3 = GATConv(self.hid*self.out_head, num_classes, heads=self.out_head, concat=False, dropout=self.dropout)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self,x, edge_index):
        
        # Dropout before the GAT layer is used to avoid overfitting
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        h = self.conv2(x, edge_index)
        x = F.elu(x+h)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        x = global_add_pool(x, torch.zeros(x.size(0), dtype=torch.long).to(self.device))
        return x,F.log_softmax(x, dim=1)