# custom_grid_search.py
import itertools
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer
from datasets import load_dataset

model_name = "distilbert-base-uncased"
dataset = load_dataset("glue", "mrpc")
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_function(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)

encoded_dataset = dataset.map(preprocess_function, batched=True)

learning_rates = [1e-5, 2e-5, 3e-5]
num_epochs = [2, 3]
best_loss = float('inf')
best_params = None

for lr, epoch in itertools.product(learning_rates, num_epochs):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        per_device_train_batch_size=16,
        learning_rate=lr,
        num_train_epochs=epoch,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
    )
    metrics = trainer.evaluate()
    eval_loss = metrics["eval_loss"]
    print(f"LR: {lr}, Epochs: {epoch}, Eval Loss: {eval_loss}")
    if eval_loss < best_loss:
        best_loss = eval_loss
        best_params = {"learning_rate": lr, "num_train_epochs": epoch}

print("Best hyperparameters found:", best_params)
