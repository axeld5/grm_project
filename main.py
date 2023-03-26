import torch
import pandas as pd 
import spacy

from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

from load_datasets import load_dataset, preprocess_dataset, remove_too_small, get_num_classes
from models.gat import GAT
from models.gcn import GCN 
from models.graphsage import GraphSAGE
from train_test_text_models import train, test

if __name__ == "__main__":    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #load_dataset
    dataset_name = "amazon"
    df = load_dataset(dataset_name)
    amount_taken = 3000
    list_of_reviews = remove_too_small(preprocess_dataset(df, amount_taken))
    num_classes = get_num_classes(dataset_name)
    train_data, test_data = train_test_split(list_of_reviews, test_size=0.1)

    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    num_node_features = 300 
    epochs = 10

    model = GCN(num_node_features, 256, num_classes, dropout=0.2).to(device)
    for epoch in range(epochs):
        loss = train(model, train_loader, device)
        train_acc = test(model, train_loader, device)
        test_acc = test(model, test_loader, device)
        print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}", f"Test Acc: {test_acc:.4f}")
    print(f'\nGCN test accuracy: {test(model, test_loader, device)*100:.2f}%\n')

    model = GAT(num_node_features, 128, num_classes, dropout=0.2).to(device)
    for epoch in range(epochs):
        loss = train(model, train_loader, device)
        train_acc = test(model, train_loader, device)
        test_acc = test(model, test_loader, device)
        print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}", f"Test Acc: {test_acc:.4f}")
    print(f'\nGAT test accuracy: {test(model, test_loader, device)*100:.2f}%\n')

    model = GraphSAGE(num_node_features, 300, num_classes).to(device)
    for epoch in range(epochs):
        loss = train(model, train_loader, device)
        train_acc = test(model, train_loader, device)
        test_acc = test(model, test_loader, device)
        print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}", f"Test Acc: {test_acc:.4f}")
    print(f'\nGraphSage test accuracy: {test(model, test_loader, device)*100:.2f}%\n')