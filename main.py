import torch
from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd 
import spacy

from preprocessing_files.preprocess_review import preprocess_review
from preprocessing_files.utils import get_num_pos_neg
from models.simple_graphnet import Classifier
from models.gat import GAT 
from models.gcn import GCN 
from models.graphsage import GraphSAGE
from train_test_text_models import train, test

if __name__ == "__main__":    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #load_dataset
    df = pd.read_csv('preprocessing_files/data/train.csv')
    df = df.sample(frac=1, random_state = 42).reset_index(drop=True)
    df.head()
    nlp = spacy.load('en_core_web_md')

    list_of_reviews = [] 

    amount_taken = 200 #must be inferior to 2500
    frac_taken = amount_taken//10
    for i in range(amount_taken):
        data = preprocess_review(df['review'][i], df['label'][i], nlp)
        list_of_reviews.append(data)
        if i%frac_taken == 0:
            print(i)
    print(list_of_reviews[0])

    train_data, test_data = train_test_split(list_of_reviews, test_size=0.2, random_state=42)

    pos,neg = get_num_pos_neg(train_data)

    print(f'Number of positive reviews in the training set: {pos}')
    print(f'Number of negative reviews in the training set: {neg}')
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    num_node_features = 300 
    num_classes = 2
    epochs = 30
    model = Classifier(num_node_features, 128, num_classes).to(device)
    for epoch in range(epochs):
        loss = train(model, train_loader, device)
        train_acc = test(model, train_loader, device)
        test_acc = test(model, test_loader, device)
        print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}", f"Test Acc: {test_acc:.4f}")
    print(f'\nSimple model test accuracy: {test(model, test_loader, device)*100:.2f}%\n')

    model = GCN(num_node_features, num_classes).to(device)
    for epoch in range(epochs):
        loss = train(model, train_loader, device)
        train_acc = test(model, train_loader, device)
        test_acc = test(model, test_loader, device)
        print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}", f"Test Acc: {test_acc:.4f}")
    print(f'\nGCN test accuracy: {test(model, test_loader, device)*100:.2f}%\n')

    model = GAT(num_node_features, num_classes).to(device)
    for epoch in range(epochs):
        loss = train(model, train_loader, device)
        train_acc = test(model, train_loader, device)
        test_acc = test(model, test_loader, device)
        print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}", f"Test Acc: {test_acc:.4f}")
    print(f'\nGAT test accuracy: {test(model, test_loader, device)*100:.2f}%\n')

    model = GraphSAGE(num_node_features, 64, num_classes).to(device)
    for epoch in range(epochs):
        loss = train(model, train_loader, device)
        train_acc = test(model, train_loader, device)
        test_acc = test(model, test_loader, device)
        print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}", f"Test Acc: {test_acc:.4f}")
    print(f'\nGraphSage test accuracy: {test(model, test_loader, device)*100:.2f}%\n')