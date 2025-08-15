import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.metrics import precision_recall_fscore_support
from datasets import Dataset
from src.data.preprocess import load_and_preprocess
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

def prepare_dataset(X, y, tokenizer, max_length=128):
    try:
        # Tokenize inputs
        encodings = tokenizer(X.tolist(), truncation=True, padding=True, max_length=max_length)
        
        # Create dictionary for Dataset
        data_dict = {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': y.values.tolist()  # Keep labels as [num_samples, 7]
        }
        
        # Create Hugging Face Dataset
        dataset = Dataset.from_dict(data_dict)
        logger.info(f"Dataset prepared with {len(dataset)} samples, label shape: {y.shape}")
        return dataset
    except Exception as e:
        logger.error(f"Error preparing dataset: {str(e)}")
        raise

def compute_metrics(eval_pred):
    try:
        predictions, labels = eval_pred
        predictions = (predictions > 0).astype(int)  # Convert logits to binary predictions
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    except Exception as e:
        logger.error(f"Error computing metrics: {str(e)}")
        raise

def train_model():
    # Load and preprocess data
    logger.info("Loading preprocessed data...")
    try:
        X_train_en, y_train_en, X_val_en, y_val_en, X_test_en, y_test_en = load_and_preprocess()
    except Exception as e:
        logger.error(f"Error loading preprocessed data: {str(e)}")
        raise

    # Initialize tokenizer and model
    logger.info("Initializing tokenizer and model...")
    try:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=7,  # toxic, severe_toxic, obscene, threat, insult, identity_hate, neutral
            problem_type="multi_label_classification"  # Specify multi-label classification
        ).to(device)
    except Exception as e:
        logger.error(f"Error initializing model or tokenizer: {str(e)}")
        raise

    # Prepare datasets
    logger.info("Preparing datasets...")
    try:
        train_dataset = prepare_dataset(X_train_en, y_train_en, tokenizer)
        val_dataset = prepare_dataset(X_val_en, y_val_en, tokenizer)
        test_dataset = prepare_dataset(X_test_en, y_test_en, tokenizer)
    except Exception as e:
        logger.error(f"Error preparing datasets: {str(e)}")
        raise

    # Define training arguments
    logger.info("Setting up training arguments...")
    try:
        training_args = TrainingArguments(
            output_dir='./models',
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )
    except Exception as e:
        logger.error(f"Error setting up training arguments: {str(e)}")
        raise

    # Initialize data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Initialize trainer
    logger.info("Initializing trainer...")
    try:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )
    except Exception as e:
        logger.error(f"Error initializing trainer: {str(e)}")
        raise

    # Train model
    logger.info("Training model...")
    try:
        trainer.train()
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

    # Evaluate on test set
    logger.info("Evaluating model on test set...")
    try:
        test_results = trainer.evaluate(test_dataset)
        logger.info(f"Test results: {test_results}")
    except Exception as e:
        logger.error(f"Error evaluating test set: {str(e)}")
        raise

    # Save model
    logger.info("Saving model...")
    try:
        os.makedirs('models', exist_ok=True)
        model.save_pretrained('models/bert-toxic-detector')
        tokenizer.save_pretrained('models/bert-toxic-detector')
        logger.info("Model saved to models/bert-toxic-detector")
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise

if __name__ == "__main__":
    train_model()