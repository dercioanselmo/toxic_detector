import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import precision_recall_fscore_support
from src.data.preprocess import load_and_preprocess
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def prepare_dataset(X, y, tokenizer, max_length=128):
    encodings = tokenizer(X.tolist(), truncation=True, padding=True, max_length=max_length, return_tensors='pt')
    labels = torch.tensor(y.values, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(
        encodings['input_ids'], encodings['attention_mask'], labels
    )
    return dataset

def train_model():
    # Load and preprocess data
    print("Loading preprocessed data...")
    X_train_en, y_train_en, X_val_en, y_val_en, X_test_en, y_test_en = load_and_preprocess()

    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # English-only model
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=7  # toxic, severe_toxic, obscene, threat, insult, identity_hate, neutral
    ).to(device)

    # Prepare datasets
    print("Preparing datasets...")
    train_dataset = prepare_dataset(X_train_en, y_train_en, tokenizer)
    val_dataset = prepare_dataset(X_val_en, y_val_en, tokenizer)
    test_dataset = prepare_dataset(X_test_en, y_test_en, tokenizer)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./models',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=lambda eval_pred: {
            'precision': precision_recall_fscore_support(
                eval_pred.label_ids, (eval_pred.predictions > 0).astype(int), average='weighted'
            )[0],
            'recall': precision_recall_fscore_support(
                eval_pred.label_ids, (eval_pred.predictions > 0).astype(int), average='weighted'
            )[1],
            'f1': precision_recall_fscore_support(
                eval_pred.label_ids, (eval_pred.predictions > 0).astype(int), average='weighted'
            )[2]
        }
    )

    # Train model
    print("Training model...")
    trainer.train()

    # Save model
    os.makedirs('models', exist_ok=True)
    model.save_pretrained('models/bert-toxic-detector')
    tokenizer.save_pretrained('models/bert-toxic-detector')
    print("Model saved to models/bert-toxic-detector")

if __name__ == "__main__":
    train_model()