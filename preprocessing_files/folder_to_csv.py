### imports 
import os
import csv
import pandas as pd


## function 

def folder_to_csv(folder_path):
    dico = {'neg': 0, 'pos': 1}
    pd_for_csv = pd.DataFrame()
    for i in dico.keys():

        path = os.path.join(folder_path, i)
        for file in os.listdir(path):
            if file.endswith('.txt'):
                file_path = os.path.join(path, file)
                pd_for_csv = pd_for_csv.append({'filename': open(file_path, 'r').read(), 'label': dico[i]}, ignore_index=True)
    
    pd_for_csv.to_csv(f'{folder_path}.csv', index=False)
    
    
    


### main
if __name__ == "__main__":
    folders_names = ['train', 'test']
    for path in folders_names :
        print(f'csv for {path} folder')
        folder_to_csv(path)

    print("Done!")
