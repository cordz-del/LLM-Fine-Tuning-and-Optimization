# data_preprocessing.py
import pandas as pd
from transformers import AutoTokenizer

def preprocess_data(csv_path, text_column):
    # Load dataset
    df = pd.read_csv(csv_path)
    # Drop rows with missing text
    df = df.dropna(subset=[text_column])
    # Clean text (example: lower-case)
    df[text_column] = df[text_column].str.lower().str.strip()
    return df

def tokenize_data(texts, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Tokenize texts with padding and truncation
    tokens = tokenizer(texts.tolist(), padding=True, truncation=True, return_tensors="pt")
    return tokens

if __name__ == "__main__":
    df = preprocess_data("data/dataset.csv", "text")
    tokens = tokenize_data(df["text"])
    print("Tokenized input shape:", tokens.input_ids.shape)
