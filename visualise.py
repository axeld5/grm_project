import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 

from typing import Dict 

def show_boxplot(model_dict:Dict[str, float], x:str, y:str) -> None:
    df = pd.DataFrame.from_dict(model_dict)
    sns.boxplot(x=x, y=y, data=df)
    plt.show()