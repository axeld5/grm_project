import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool, global_mean_pool

from .utils_gcn import calculate_laplacian_with_self_loop

"""
Graph Convolutional Network
GCN takes graphs as an input and applies convolution operations over the graph
"""


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim, num_classes, dropout=0.2):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
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
        # x = global_mean_pool(x, torch.zeros(x.size(0), dtype=torch.long).to(self.device))
        x = global_mean_pool(x, data.batch)
        x = self.classifier(x)
        return x, F.log_softmax(x, dim=1)


class MyGCNConv(nn.module):
    def __init__(self, input_dim, output_dim):
        super(GCN, self).__init__()
        self.input_dim = input_dim  # seq_len for prediction
        self.output_dim = output_dim  # hidden_dim for prediction
        self.weights = nn.Parameter(torch.FloatTensor(self.input_dim, self.output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights, gain=nn.init.calculate_gain("tanh"))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        num_nodes = edge_index.shape[0]
        laplacian = calculate_laplacian_with_self_loop(torch.FloatTensor(edge_index))
        # (batch_size, seq_len, num_nodes)
        batch_size = x.shape[0]
        # (num_nodes, batch_size, seq_len)
        x = x.transpose(0, 2).transpose(1, 2)
        # (num_nodes, batch_size * seq_len)
        x = x.reshape((num_nodes, batch_size * self.input_dim))
        # AX (num_nodes, batch_size * seq_len)
        ax = laplacian @ x
        # (num_nodes, batch_size, seq_len)
        ax = ax.reshape((num_nodes, batch_size, self.input_dim))
        # (num_nodes * batch_size, seq_len)
        ax = ax.reshape((num_nodes * batch_size, self.input_dim))
        # act(AXW) (num_nodes * batch_size, output_dim)
        outputs = torch.tanh(ax @ self.weights)
        # (num_nodes, batch_size, output_dim)
        outputs = outputs.reshape((num_nodes, batch_size, self.output_dim))
        # (batch_size, num_nodes, output_dim)
        outputs = outputs.transpose(0, 1)

        return outputs


class MyGCN(nn.module):
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
        # x = global_mean_pool(x, torch.zeros(x.size(0), dtype=torch.long).to(self.device))
        x = global_mean_pool(x, data.batch)
        x = self.classifier(x)
        return x, F.log_softmax(x, dim=1)
