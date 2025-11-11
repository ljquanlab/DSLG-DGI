import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
import numpy as np

MODEL_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext" 
data_files = {"train": "../data/finetune_EEtrain.jsonl", "test": "../data/finetune_EEtest.jsonl"}
raw_datasets = load_dataset("json", data_files=data_files)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=256)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch") 
train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["test"]

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

training_args = TrainingArguments(
    output_dir="./pubmedbert_finetuned_relation_classifier",
    num_train_epochs=3, 
    per_device_train_batch_size=24, 
    per_device_eval_batch_size=48, 
    learning_rate=3e-5, 
    weight_decay=0.01,
    eval_strategy="no",
    save_strategy="epoch",
    report_to="none", 
    load_best_model_at_end=False,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer, 
    data_collator=data_collator,
)

train_result = trainer.train()

final_output_dir = "./pubmedbert_finetuned_relation_classifier/final"
trainer.save_model(final_output_dir)
print("Finetuning complete. Model saved to final directory.")