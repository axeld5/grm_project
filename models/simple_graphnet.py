import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_add_pool

class GraphNet(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GraphNet, self).__init__(aggr="add")
        self.lin = torch.nn.Linear(in_channels, out_channels)
        

    def forward(self, x, edge_index):
        x = self.lin(x)
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return aggr_out

class Classifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Classifier, self).__init__()
        self.conv1 = GraphNet(in_channels, hidden_channels)
        self.conv2 = GraphNet(hidden_channels, out_channels)
        self.optimizer = torch.optim.Adam(self.parameters(),
                                      lr=0.01,
                                      weight_decay=5e-4)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        x = global_add_pool(x, torch.zeros(x.size(0), dtype=torch.long).to(self.device))
        return x, F.log_softmax(x, dim=1)