import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool

'''
Graph Convolutional Network
GCN takes graphs as an input and applies convolution operations over the graph
'''
class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim, num_classes, dropout=0.2):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.conv4 = GCNConv(hidden_dim, num_classes)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001, weight_decay=5e-4)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dropout = dropout

    def forward(self, x, edge_index):
        # x: Node feature matrix 
        # edge_index: Graph connectivity matrix 
        #x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv4(x, edge_index)
        x = global_add_pool(x, torch.zeros(x.size(0), dtype=torch.long).to(self.device))
        return x, F.log_softmax(x, dim=1)
