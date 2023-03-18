import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_add_pool, global_mean_pool  

'''
Graph SAGE: SAmpling and aggreGatE, 
Samples only a subset of neighboring nodes at different depth layers, 
and then the aggregator takes neighbors of the previous layers and aggregates them
'''
class GraphSAGE(torch.nn.Module):
  """GraphSAGE"""
  def __init__(self, num_node_features, hidden_dim, num_classes):
    super().__init__()
    self.sage1 = SAGEConv(num_node_features, hidden_dim*2)
    self.sage2 = SAGEConv(hidden_dim*2, hidden_dim)
    self.sage3 = SAGEConv(hidden_dim, hidden_dim)
    self.sage4 = SAGEConv(hidden_dim, num_classes)
    self.optimizer = torch.optim.Adam(self.parameters(),
                                      lr=0.0001)
                                        # weight_decay=5e-4)
                                      
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  def forward(self, x, edge_index):
    ## layer 1 
    h = self.sage1(x, edge_index)
    h = torch.relu(h)
    h = F.dropout(h, p=0.2, training=self.training)

    ## layer 2

    h = self.sage2(h, edge_index)
    h = torch.relu(h)
    h = F.dropout(h, p=0.2, training=self.training)

    # layer 3 
    h = self.sage3(h, edge_index)
    h = torch.relu(h)
    h = F.dropout(h, p=0.5, training=self.training)

     # layer 4
    h = self.sage3(h, edge_index)
    h = torch.relu(h)
    h = F.dropout(h, p=0.5, training=self.training)

     # layer 5
    h = self.sage3(h, edge_index)
    h = torch.relu(h)
    h = F.dropout(h, p=0.2, training=self.training)

    ## layer 6
    h = self.sage4(h, edge_index)
    h = global_mean_pool(h, torch.zeros(h.size(0), dtype=torch.long).to(self.device))
    return h, F.log_softmax(h, dim=1)