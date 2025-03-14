# optuna_tuning.py
import optuna
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

model_name = "distilbert-base-uncased"
dataset = load_dataset("glue", "mrpc")
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_function(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)

encoded_dataset = dataset.map(preprocess_function, batched=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=16,
)

def objective(trial):
    training_args.learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 5e-5)
    training_args.num_train_epochs = trial.suggest_int("num_train_epochs", 2, 4)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
    )
    metrics = trainer.evaluate()
    return metrics["eval_loss"]

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=5)
print("Best trial:", study.best_trial.params)
