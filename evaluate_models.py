from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

from models.gat import GAT, EdgeGAT
from models.gcn import GCN
from models.graphsage import GraphSAGE
from train_test_text_models import train, test, edge_train, edge_test

def evaluate_models(list_of_reviews, num_node_features, num_classes, device, epochs=10, 
                    gcn_hidden_size=256, gat_hidden_size=128, gsage_hidden_size=300, num_loops=5):
    perf_dict = {"model_name":[], "accuracy":[]}
    for _ in range(num_loops):
        train_data, test_data = train_test_split(list_of_reviews, test_size=0.1)
        train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False)        
        final_gcn_accuracy, final_gat_accuracy, final_gsage_accuracy = evaluate_loop(train_loader, test_loader,
                                            num_node_features, num_classes, device, epochs, gcn_hidden_size, gat_hidden_size, gsage_hidden_size)
        perf_dict["model_name"].append("gcn")
        perf_dict["accuracy"].append(final_gcn_accuracy)
        perf_dict["model_name"].append("gat")
        perf_dict["accuracy"].append(final_gat_accuracy)
        perf_dict["model_name"].append("graphsage")
        perf_dict["accuracy"].append(final_gsage_accuracy)
    return perf_dict

def evaluate_edge_model(list_of_reviews, num_node_features, num_classes, device, epochs=10, 
                    gcn_hidden_size=256, gat_hidden_size=128, gsage_hidden_size=300, num_loops=5):
    perf_dict = {"model_name":[], "accuracy":[]}
    for _ in range(num_loops):
        train_data, test_data = train_test_split(list_of_reviews, test_size=0.1)
        train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False)        
        final_gat_accuracy, final_edge_gat_accuracy = evaluate_edge_loop(train_loader, test_loader,
                                            num_node_features, num_classes, device, epochs, gcn_hidden_size, gat_hidden_size, gsage_hidden_size)
        perf_dict["model_name"].append("gat")
        perf_dict["accuracy"].append(final_gat_accuracy)
        perf_dict["model_name"].append("graphsage")
        perf_dict["accuracy"].append(final_edge_gat_accuracy)
    return perf_dict

def evaluate_loop(train_loader, test_loader, num_node_features, num_classes, device, epochs=10, gcn_hidden_size=256, gat_hidden_size=128, gsage_hidden_size=300):
    
    gcn_model = GCN(num_node_features, gcn_hidden_size, num_classes, dropout=0.2).to(device)
    for epoch in range(epochs):
        loss = train(gcn_model, train_loader, device)
        train_acc = test(gcn_model, train_loader, device)
        test_acc = test(gcn_model, test_loader, device)
        print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}", f"Test Acc: {test_acc:.4f}")
    final_gcn_accuracy = test(gcn_model, test_loader, device)*100
    print(f'\nGCN test accuracy: {final_gcn_accuracy:.2f}%\n')

    gat_model = GAT(num_node_features, gat_hidden_size, num_classes, dropout=0.2).to(device)
    for epoch in range(epochs):
        loss = train(gat_model, train_loader, device)
        train_acc = test(gat_model, train_loader, device)
        test_acc = test(gat_model, test_loader, device)
        print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}", f"Test Acc: {test_acc:.4f}")
    final_gat_accuracy = test(gat_model, test_loader, device)*100
    print(f'\nGAT test accuracy: {final_gat_accuracy:.2f}%\n')

    gsage_model = GraphSAGE(num_node_features, gsage_hidden_size, num_classes).to(device)
    for epoch in range(epochs):
        loss = train(gsage_model, train_loader, device)
        train_acc = test(gsage_model, train_loader, device)
        test_acc = test(gsage_model, test_loader, device)
        print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}", f"Test Acc: {test_acc:.4f}")
    final_gsage_accuracy = test(gsage_model, test_loader, device)*100
    print(f'\nGraphSage test accuracy: {final_gsage_accuracy:.2f}%\n')

    return final_gcn_accuracy, final_gat_accuracy, final_gsage_accuracy

def evaluate_edge_loop(train_loader, test_loader, num_node_features, num_classes, device, epochs=10, gcn_hidden_size=256, gat_hidden_size=128, gsage_hidden_size=300):

    gat_model = GAT(num_node_features, gat_hidden_size, num_classes, dropout=0.2).to(device)
    for epoch in range(epochs):
        loss = train(gat_model, train_loader, device)
        train_acc = test(gat_model, train_loader, device)
        test_acc = test(gat_model, test_loader, device)
        print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}", f"Test Acc: {test_acc:.4f}")
    final_gat_accuracy = test(gat_model, test_loader, device)*100
    print(f'\nGAT test accuracy: {final_gat_accuracy:.2f}%\n')

    edge_gat_model = EdgeGAT(num_node_features, gat_hidden_size, num_classes, dropout=0.2).to(device)
    for epoch in range(epochs):
        loss = edge_train(edge_gat_model, train_loader, device)
        train_acc = edge_test(edge_gat_model, train_loader, device)
        test_acc = edge_test(edge_gat_model, test_loader, device)
        print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}", f"Test Acc: {test_acc:.4f}")
    final_edge_gat_accuracy = edge_test(edge_gat_model, test_loader, device)*100
    print(f'\nEdge GAT test accuracy: {final_edge_gat_accuracy:.2f}%\n')

    return final_gat_accuracy, final_edge_gat_accuracy