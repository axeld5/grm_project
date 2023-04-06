import numpy as np

from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

from models.gat import GAT, EdgeGAT
from models.gcn import GCN
from models.mygcn import MyGCN
from models.graphsage import GraphSAGE
from models.mygat import MyGraphGAT
from train_test_text_models import train, test, edge_train, edge_test


def evaluate_models(list_of_reviews, num_node_features, num_classes, device, epochs=10, 
                    gcn_hidden_size=256, gat_hidden_size=128, gsage_hidden_size=300, num_loops=5, batch_size=1):
    perf_dict = {"model_name": [], "accuracy": []}
    history_dict = {
        "gcn": np.zeros((num_loops, epochs)),
        "gcn (our)": np.zeros((num_loops, epochs)),
        "gat": np.zeros((num_loops, epochs)),
        "gat (our)": np.zeros((num_loops, epochs)),
        "graphsage": np.zeros((num_loops, epochs)),
    }
    for i in range(num_loops):
        train_data, test_data = train_test_split(list_of_reviews, test_size=0.1)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        (final_gcn_accuracy, gcn_history, final_gat_accuracy, gat_history, final_gsage_accuracy, gsage_history, 
        final_my_gcn_accuracy, my_gcn_history, final_my_gat_accuracy, my_gat_history) = evaluate_loop(
            train_loader, test_loader, num_node_features, num_classes, device, epochs, gcn_hidden_size, gat_hidden_size, gsage_hidden_size,
        )
        perf_dict["model_name"].append("gcn")
        perf_dict["accuracy"].append(final_gcn_accuracy)
        perf_dict["model_name"].append("gat")
        perf_dict["accuracy"].append(final_gat_accuracy)
        perf_dict["model_name"].append("graphsage")
        perf_dict["accuracy"].append(final_gsage_accuracy)
        perf_dict["model_name"].append("gcn (our)")
        perf_dict["accuracy"].append(final_my_gcn_accuracy)
        perf_dict["model_name"].append("gat (our)")
        perf_dict["accuracy"].append(final_my_gat_accuracy)
        history_dict["gcn"][i, :] = gcn_history
        history_dict["gat"][i, :] = gat_history
        history_dict["graphsage"][i, :] = gsage_history
        history_dict["gcn (our)"][i, :] = my_gcn_history
        history_dict["gat (our)"][i, :] = my_gat_history
    return perf_dict, history_dict


def evaluate_loop(train_loader, test_loader, num_node_features, num_classes, device, epochs=10, gcn_hidden_size=256, gat_hidden_size=128, gsage_hidden_size=300):

    gcn_model = GCN(num_node_features, gcn_hidden_size, num_classes, dropout=0.2).to(device)
    gcn_history = np.zeros(epochs)
    for epoch in range(epochs):
        loss = train(gcn_model, train_loader, device)
        train_acc = test(gcn_model, train_loader, device)
        test_acc = test(gcn_model, test_loader, device)
        print(
            f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}",
            f"Test Acc: {test_acc:.4f}",
        )
        gcn_history[epoch] = test_acc
    final_gcn_accuracy = test(gcn_model, test_loader, device) * 100
    print(f"\nGCN test accuracy: {final_gcn_accuracy:.2f}%\n")    

    gat_model = GAT(num_node_features, gat_hidden_size, num_classes, dropout=0.2).to(device)
    gat_history = np.zeros(epochs)
    for epoch in range(epochs):
        loss = train(gat_model, train_loader, device)
        train_acc = test(gat_model, train_loader, device)
        test_acc = test(gat_model, test_loader, device)
        print(
            f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}",
            f"Test Acc: {test_acc:.4f}",
        )
        gat_history[epoch] = test_acc
    final_gat_accuracy = test(gat_model, test_loader, device) * 100
    print(f"\nGAT test accuracy: {final_gat_accuracy:.2f}%\n")

    gsage_model = GraphSAGE(num_node_features, gsage_hidden_size, num_classes).to(device)
    gsage_history = np.zeros(epochs)
    for epoch in range(epochs):
        loss = train(gsage_model, train_loader, device)
        train_acc = test(gsage_model, train_loader, device)
        test_acc = test(gsage_model, test_loader, device)
        print(
            f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}",
            f"Test Acc: {test_acc:.4f}",
        )
        gsage_history[epoch] = test_acc
    final_gsage_accuracy = test(gsage_model, test_loader, device) * 100
    print(f"\nGraphSage test accuracy: {final_gsage_accuracy:.2f}%\n")

    my_gcn_model = MyGCN(num_node_features, gcn_hidden_size, num_classes, dropout=0.2).to(device)
    my_gcn_history = np.zeros(epochs)
    for epoch in range(epochs):
        loss = train(my_gcn_model, train_loader, device)
        train_acc = test(my_gcn_model, train_loader, device)
        test_acc = test(my_gcn_model, test_loader, device)
        print(
            f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}",
            f"Test Acc: {test_acc:.4f}",
        )
        my_gcn_history[epoch] = test_acc
    final_my_gcn_accuracy = test(my_gcn_model, test_loader, device) * 100
    print(f"\n(Our) GCN test accuracy: {final_my_gcn_accuracy:.2f}%\n")

    my_gat_model = MyGraphGAT(num_node_features, gat_hidden_size, num_classes, dropout=0.2).to(device)
    my_gat_history = np.zeros(epochs)
    for epoch in range(epochs):
        loss = train(my_gat_model, train_loader, device)
        train_acc = test(my_gat_model, train_loader, device)
        test_acc = test(my_gat_model, test_loader, device)
        print(
            f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}",
            f"Test Acc: {test_acc:.4f}",
        )
        my_gat_history[epoch] = test_acc
    final_my_gat_accuracy = test(my_gat_model, test_loader, device) * 100
    print(f"\n(Our) GAT test accuracy: {final_my_gat_accuracy:.2f}%\n")

    return (
        final_gcn_accuracy,
        gcn_history,
        final_gat_accuracy,
        gat_history,
        final_gsage_accuracy,
        gsage_history,
        final_my_gcn_accuracy,
        my_gcn_history,
        final_my_gat_accuracy,
        my_gat_history
    )


def evaluate_edge_model(list_of_reviews, num_node_features, num_classes, device, epochs=10,
                        gcn_hidden_size=256, gat_hidden_size=128, gsage_hidden_size=300, num_loops=5, batch_size=1):
    perf_dict = {"model_name": [], "accuracy": []}
    history_dict = {
        "gat": np.zeros((num_loops, epochs)),
        "directed_edgegat": np.zeros((num_loops, epochs)),
    }
    for i in range(num_loops):
        train_data, test_data = train_test_split(list_of_reviews, test_size=0.1)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        edge_dim = 38
        final_gat_accuracy, gat_history, final_edge_gat_accuracy, di_edge_gat_history = evaluate_edge_loop(train_loader, test_loader,
            num_node_features, num_classes, edge_dim, device, epochs, gcn_hidden_size, gat_hidden_size, gsage_hidden_size)
        perf_dict["model_name"].append("gat")
        perf_dict["accuracy"].append(final_gat_accuracy)
        perf_dict["model_name"].append("directed_edgegat")
        perf_dict["accuracy"].append(final_edge_gat_accuracy)
        history_dict["gat"][i, :] = gat_history
        history_dict["directed_edgegat"][i, :] = di_edge_gat_history
    return perf_dict, history_dict


def evaluate_all_edge_models(di_list_of_reviews, undi_list_of_reviews, wtd_list_of_reviews, num_node_features,
    num_classes, device, epochs=10, gat_hidden_size=128, num_loops=5, batch_size=1):
    perf_dict = {"model_name": [], "accuracy": []}
    history_dict = {
        "gat": np.zeros((num_loops, epochs)),
        "directed_edgegat": np.zeros((num_loops, epochs)),
        "undirected_edgegat": np.zeros((num_loops, epochs)),
        "weighted_edgegat": np.zeros((num_loops, epochs)),
    }
    seed_list = [13, 42, 200, 500, 2000]
    for i in range(num_loops):
        di_train_data, di_test_data = train_test_split(
            di_list_of_reviews, test_size=0.1, random_state=seed_list[i]
        )
        di_train_loader = DataLoader(di_train_data, batch_size=batch_size, shuffle=True)
        di_test_loader = DataLoader(di_test_data, batch_size=batch_size, shuffle=False)
        edge_dim = 38
        final_gat_accuracy, gat_history, final_di_edge_gat_accuracy, di_edge_gat_history = evaluate_edge_loop(di_train_loader, di_test_loader,
            num_node_features, num_classes, edge_dim, device, epochs, gat_hidden_size)
        perf_dict["model_name"].append("gat")
        perf_dict["accuracy"].append(final_gat_accuracy)
        perf_dict["model_name"].append("directed_edgegat")
        perf_dict["accuracy"].append(final_di_edge_gat_accuracy)

        undi_train_data, undi_test_data = train_test_split(undi_list_of_reviews, test_size=0.1, random_state=seed_list[i])
        undi_train_loader = DataLoader(undi_train_data, batch_size=batch_size, shuffle=True)
        undi_test_loader = DataLoader(undi_test_data, batch_size=batch_size, shuffle=False)
        edge_dim = 19
        _, _, final_undi_edge_gat_accuracy, undi_edge_gat_history = evaluate_edge_loop(undi_train_loader, undi_test_loader,
            num_node_features, num_classes, edge_dim, device, epochs, gat_hidden_size, train_gat=False)
        perf_dict["model_name"].append("undirected_edgegat")
        perf_dict["accuracy"].append(final_undi_edge_gat_accuracy)

        wtd_train_data, wtd_test_data = train_test_split(wtd_list_of_reviews, test_size=0.1, random_state=seed_list[i])
        wtd_train_loader = DataLoader(wtd_train_data, batch_size=batch_size, shuffle=True)
        wtd_test_loader = DataLoader(wtd_test_data, batch_size=batch_size, shuffle=False)
        edge_dim = 1
        _, _, final_wtd_edge_gat_accuracy, wtd_edge_gat_history = evaluate_edge_loop(wtd_train_loader, wtd_test_loader,
            num_node_features, num_classes, edge_dim, device, epochs, gat_hidden_size, train_gat=False)
        perf_dict["model_name"].append("weighted_edgegat")
        perf_dict["accuracy"].append(final_wtd_edge_gat_accuracy)

        history_dict["gat"][i, :] = gat_history
        history_dict["directed_edgegat"][i, :] = di_edge_gat_history
        history_dict["undirected_edgegat"][i, :] = undi_edge_gat_history
        history_dict["weighted_edgegat"][i, :] = wtd_edge_gat_history
    return perf_dict, history_dict


def evaluate_edge_loop(train_loader,test_loader, num_node_features, 
                       num_classes, edge_dim, device, epochs=10, gat_hidden_size=128, train_gat=True):
    gat_history = np.zeros(epochs)
    if train_gat:
        gat_model = GAT(num_node_features, gat_hidden_size, num_classes, dropout=0.2).to(device)
        for epoch in range(epochs):
            loss = train(gat_model, train_loader, device)
            train_acc = test(gat_model, train_loader, device)
            test_acc = test(gat_model, test_loader, device)
            print(
                f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}",
                f"Test Acc: {test_acc:.4f}",
            )
            gat_history[epoch] = test_acc
        final_gat_accuracy = test(gat_model, test_loader, device) * 100
        print(f"\nGAT test accuracy: {final_gat_accuracy:.2f}%\n")
    else:
        final_gat_accuracy = 0

    edge_gat_model = EdgeGAT(num_node_features, gat_hidden_size, num_classes, edge_dim, dropout=0.2).to(device)
    edge_gat_history = np.zeros(epochs)
    for epoch in range(epochs):
        loss = edge_train(edge_gat_model, train_loader, device)
        train_acc = edge_test(edge_gat_model, train_loader, device)
        test_acc = edge_test(edge_gat_model, test_loader, device)
        print(
            f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}",
            f"Test Acc: {test_acc:.4f}",
        )
        edge_gat_history[epoch] = test_acc
    final_edge_gat_accuracy = edge_test(edge_gat_model, test_loader, device) * 100
    print(f"\nEdge GAT test accuracy: {final_edge_gat_accuracy:.2f}%\n")

    return final_gat_accuracy, gat_history, final_edge_gat_accuracy, edge_gat_history
