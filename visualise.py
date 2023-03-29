import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np

from typing import Dict 

def show_boxplot(perf_dict:Dict[str, float], x:str, y:str) -> None:
    df = pd.DataFrame.from_dict(perf_dict)
    sns.boxplot(x=x, y=y, data=df)
    plt.show()

def compare_models(history_dict:Dict[str, float], n_epochs:int, num_loops:int) -> None:
    model_names = list(history_dict.keys())
    fig = plt.figure()
    x = np.arange(n_epochs) + np.ones(n_epochs)
    for model_name in model_names:
        avgd_score = np.mean(np.array(history_dict[model_name]), axis=0)
        plt.plot(x, avgd_score, label=model_name)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy Score")
    plt.title("Average evolution of the models' validation accuracies over "+str(num_loops)+" runs")
    plt.legend(model_names)
    plt.show()

def get_avg_std(history_dict:Dict[str, np.ndarray]) -> Dict[str, float]:
    model_names = list(history_dict.keys())
    avg_std_dict = {}
    for model_name in model_names: 
        avg_std_dict[model_name] = np.zeros(2)
        avg_std_dict[model_name][0] = np.mean(history_dict[model_name], axis=0)[-1]
        avg_std_dict[model_name][1] = np.std(history_dict[model_name], axis=0)[-1]
    return avg_std_dict