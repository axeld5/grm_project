import torch.nn.functional as F 
import torch 

def train(model, loader, device):
    model.train()
    optimizer = model.optimizer
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        _, out = model(data.x.float(), data.edge_index)
        loss = F.nll_loss(out, data.y.long())
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / len(loader.dataset)
    
    
# Define the testing loop
def test(model, loader, device):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        _, out = model(data.x.float(), data.edge_index)
        pred = out.argmax(dim=1)
        correct += pred.eq(data.y.long()).sum().item()
    return correct / len(loader.dataset)

def edge_train(model, loader, device):
    model.train()
    optimizer = model.optimizer
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        _, out = model(data.x.float(), data.edge_index, edge_weight = data.edge_attr)
        loss = F.nll_loss(out, data.y.long())
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / len(loader.dataset)

def edge_test(model, loader, device):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        _, out = model(data.x.float(), data.edge_index, edge_weight = data.edge_attr)
        pred = out.argmax(dim=1)
        correct += pred.eq(data.y.long()).sum().item()
    return correct / len(loader.dataset)