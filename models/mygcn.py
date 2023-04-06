import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

from .utils_gcn import calculate_laplacian_with_self_loop

class MyGCNConv(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim 
        self.output_dim = output_dim  
        self.weights = nn.Parameter(torch.zeros(self.input_dim, self.output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights, gain=nn.init.calculate_gain("tanh"))

    def forward(self, x, edge_index):
        laplacian = calculate_laplacian_with_self_loop(torch.tensor(edge_index))
        ax = laplacian @ x
        outputs = torch.tanh(ax @ self.weights)

        return outputs


class MyGCN(nn.Module):
    def __init__(self, num_node_features, hidden_dim, num_classes, dropout=0.2):
        super().__init__()
        self.conv1 = MyGCNConv(num_node_features, hidden_dim)
        self.conv2 = MyGCNConv(hidden_dim, hidden_dim)
        self.conv3 = MyGCNConv(hidden_dim, hidden_dim)
        self.classifier = torch.nn.Linear(hidden_dim, num_classes)
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=0.0001, weight_decay=5e-4
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        h = self.conv2(x, edge_index)
        x = F.relu(x + h)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, data.batch)
        x = self.classifier(x)
        return x, F.log_softmax(x, dim=1)