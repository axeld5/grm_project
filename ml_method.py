import torch
import pandas as pd 
import evaluate
import numpy as np
import os 
os.environ["WANDB_DISABLED"] = "true"

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
  

if __name__ == "__main__":    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = pd.read_csv('preprocessing_files/data/train.csv')
    texts = df["review"].tolist()
    labels = df["label"].to_numpy()
    vectorizer = TfidfVectorizer() 
    vect_texts = vectorizer.fit_transform(texts) 
    X_train, X_test, y_train, y_test = train_test_split(vect_texts, labels)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    pred_score = accuracy_score(clf.predict(X_test), y_test)
    print("test_acc="+str(pred_score))
