# grm_project
Repository for the project for the Graphical Models course of CentraleSupélec.

# How to benchmark models.
All models are implemented in the models folder. Functions to get the dataset are implemented in load_dataset, which fetches csv files put in preprocessing file, and uses the specified tokenizing method, using the methods put in bigram_tokenize.py.

To evaluate all models implemented, run "main_eval.py", specifying the dataset name you want to use in dataset_name.
To evaluate all preprocessing methods that involve the edges, run "edge_eval.py", specifying the dataset name you want to use in dataset_name.