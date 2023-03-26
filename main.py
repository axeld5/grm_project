import torch
import pandas as pd 
import spacy

from load_datasets import load_dataset, preprocess_dataset, remove_too_small, get_num_classes
from evaluate_models import evaluate_loop, evaluate_models
from visualise import show_boxplot

if __name__ == "__main__":    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #load_dataset
    dataset_name = "imdb"
    df = load_dataset(dataset_name)
    amount_taken = 1000
    list_of_reviews = remove_too_small(preprocess_dataset(dataset_name, df, amount_taken))
    num_classes = get_num_classes(dataset_name)
    num_node_features = 300 
    epochs = 10
    perf_dict = evaluate_models(list_of_reviews, num_node_features, num_classes, device)
    show_boxplot(perf_dict, x="model_name", y="accuracy")