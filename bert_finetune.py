import torch
import numpy as np 
import pandas as pd 
import os 
os.environ["WANDB_DISABLED"] = "true"

from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer

from load_datasets import load_dataset, get_num_classes
from bert_specials import compute_metrics, from_df_to_dataset

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

if __name__ == "__main__":    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_name = "amazon"    
    num_labels = get_num_classes(dataset_name)
    df = load_dataset(dataset_name)
    data = from_df_to_dataset(dataset_name, df)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    tokenized_data = data.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    n_loops = 3 
    score_matrix = np.zeros(n_loops)
    for i in range(n_loops):
        if dataset_name == "imdb":
            id2label = {0: "NEGATIVE", 1: "POSITIVE"}
            label2id = {"NEGATIVE": 0, "POSITIVE": 1}
            model = AutoModelForSequenceClassification.from_pretrained(
                "distilbert-base-uncased", num_labels=num_labels, id2label=id2label, label2id=label2id
            )
        else: 
            model = AutoModelForSequenceClassification.from_pretrained(
                "distilbert-base-uncased", num_labels=num_labels
            )
        
        training_args = TrainingArguments(
            output_dir=".",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=1,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_data["train"],
            eval_dataset=tokenized_data["test"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        score_matrix[i] = trainer.evaluate()["eval_accuracy"]
    avg_std_matrix = np.zeros(2)
    avg_std_matrix[0] = np.mean(score_matrix)
    avg_std_matrix[1] = np.std(score_matrix)
    print(avg_std_matrix)