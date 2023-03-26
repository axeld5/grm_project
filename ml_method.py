import torch
import pandas as pd 

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from load_datasets import load_dataset, get_texts_and_labels
  

if __name__ == "__main__":    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_name = "imdb"
    texts, labels = get_texts_and_labels(dataset_name, load_dataset(dataset_name))
    vectorizer = TfidfVectorizer() 
    vect_texts = vectorizer.fit_transform(texts) 
    X_train, X_test, y_train, y_test = train_test_split(vect_texts, labels)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    pred_score = accuracy_score(clf.predict(X_test), y_test)
    print("test_acc="+str(pred_score))
