import pandas as pd 
import spacy 
import torch

from preprocessing_files.bigram_tokenize import biagram_preprocessing

def load_dataset(dataset_name:str) -> pd.DataFrame:
    df = pd.DataFrame()
    if dataset_name == "imdb":
        df = pd.read_csv('preprocessing_files/data/train.csv')
        df = df.sample(frac=1).reset_index(drop=True)
    elif dataset_name == "amazon":
        df_fashion = pd.read_csv('preprocessing_files/multi_class_data/fashion.csv')
        df_music = pd.read_csv('preprocessing_files/multi_class_data/music.csv')
        df_sport = pd.read_csv('preprocessing_files/multi_class_data/sport.csv')
        df_pet = pd.read_csv('preprocessing_files/multi_class_data/pet.csv')

        df = pd.concat([df_fashion, df_music, df_sport, df_pet], ignore_index=True)
        df = df.sample(frac=1).reset_index(drop=True)
    return df 

def preprocess_dataset(dataset_name:str, df:pd.DataFrame, amount_taken:int) -> list:
    nlp = spacy.load('en_core_web_md')
    texts, labels = get_texts_and_labels(dataset_name, df)
    list_of_reviews = [] 
    frac_taken = amount_taken//10
    for i in range(amount_taken):
        data = biagram_preprocessing(texts[i], labels[i], nlp)
        list_of_reviews.append(data)
        if i%frac_taken == 0:
            print(i)
    return list_of_reviews 

def remove_too_small(review_list:list) -> list:
    for i,review in enumerate(review_list):
        if review.edge_index.shape < torch.Size([2]):
            print(review.edge_index.shape)
            print(i)
            print(review.y)
            print(review.x.shape)
            review_list.pop(i)
    print(len(review_list))
    return review_list

def get_num_classes(dataset_name:str) -> int: 
    if dataset_name == "imdb":
        return 2 
    elif dataset_name == "amazon":
        return 4

def get_texts_and_labels(dataset_name:str, df:pd.DataFrame):
    if dataset_name == "imdb":
        texts = df["review"].tolist() 
        labels = df["label"].to_numpy()
    elif dataset_name == "amazon":
        texts = df["reviewText"].tolist()
        labels = df["label"].to_numpy()
    return texts, labels