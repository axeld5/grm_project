import evaluate
import numpy as np
import pandas as pd 

from datasets import Dataset

def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

def from_df_to_dataset(dataset_name:str, df:pd.DataFrame) -> Dataset:
    df["label"] = df["label"].astype(int)
    if dataset_name == "imdb":
        df.rename(columns={"review":"text"}, inplace=True)
    elif dataset_name == "amazon":
        df.rename(columns={"reviewText":"text"}, inplace=True)
    elif dataset_name == "newsgroup":
        df.rename(columns={"text_cleaned":"text"}, inplace=True)
        df.drop(["target"], axis=1, inplace=True)
        df["text"] = df["text"].astype('U')
    data = Dataset.from_pandas(df).train_test_split(test_size=0.1)
    return data 