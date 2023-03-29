import torch
import pandas as pd 
import spacy

from load_datasets import load_dataset, preprocess_dataset, remove_too_small, get_num_classes
from evaluate_models import evaluate_models, evaluate_edge_model
from visualise import show_boxplot, get_avg_std, compare_models

if __name__ == "__main__":    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_name = "newsgroup"
    df = load_dataset(dataset_name)
    if dataset_name == "amazon":
        amount_taken = 4000
        batch_size= 32
    elif dataset_name == "imdb":
        amount_taken = 25000
        batch_size = 256
    elif dataset_name == "newsgroup":
        amount_taken = 18828
        batch_size = 256
    list_of_reviews = remove_too_small(preprocess_dataset(dataset_name, df, amount_taken))
    num_classes = get_num_classes(dataset_name)
    num_node_features = 300 
    if dataset_name != "newsgroup":
        epochs = 20
        num_loops = 5
    else: 
        epochs = 40
        num_loops = 3
    perf_dict, history_dict = evaluate_models(list_of_reviews, num_node_features, num_classes, device, epochs=epochs, batch_size=batch_size, num_loops=num_loops)
    print(get_avg_std(history_dict))
    compare_models(history_dict, epochs, num_loops)
    show_boxplot(perf_dict, x="model_name", y="accuracy")