import torch
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from load_datasets import load_dataset, get_texts_and_labels
  

if __name__ == "__main__":    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_name = "newsgroup"
    texts, labels = get_texts_and_labels(dataset_name, load_dataset(dataset_name))
    vectorizer = TfidfVectorizer(min_df=50) 
    vect_texts = vectorizer.fit_transform(texts) 
    n_loops = 5  
    score_matrix = np.zeros(n_loops)
    for i in range(n_loops):
        X_train, X_test, y_train, y_test = train_test_split(vect_texts, labels)
        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)
        pred_score = accuracy_score(clf.predict(X_test), y_test)
        score_matrix[i] = pred_score
    avg_std_matrix = np.zeros(2)
    avg_std_matrix[0] = np.mean(score_matrix)
    avg_std_matrix[1] = np.std(score_matrix)
    print(avg_std_matrix)
