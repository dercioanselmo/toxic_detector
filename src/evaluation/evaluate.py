import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import precision_recall_fscore_support
from datasets import Dataset
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def prepare_dataset(X, y, tokenizer, max_length=128):
    try:
        encodings = tokenizer(X.tolist(), truncation=True, padding=True, max_length=max_length)
        data_dict = {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': y.values.astype(float).tolist()
        }
        dataset = Dataset.from_dict(data_dict)
        logger.info(f"Dataset prepared with {len(dataset)} samples, label shape: {y.shape}")
        return dataset
    except Exception as e:
        logger.error(f"Error preparing dataset: {str(e)}")
        raise

def compute_metrics(eval_pred):
    try:
        predictions, labels = eval_pred
        predictions = (predictions > 0).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted', zero_division=0)
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    except Exception as e:
        logger.error(f"Error computing metrics: {str(e)}")
        raise

def evaluate_model():
    try:
        # Load test data
        logger.info("Loading test data...")
        test_df = pd.read_csv('data/test_en.csv')
        X_test = test_df['comment_text']
        y_test = test_df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate', 'neutral']]

        # Initialize tokenizer and model
        logger.info("Initializing tokenizer and model...")
        tokenizer = BertTokenizer.from_pretrained('models/bert-toxic-detector')
        model = BertForSequenceClassification.from_pretrained('models/bert-toxic-detector')

        # Prepare test dataset
        logger.info("Preparing test dataset...")
        test_dataset = prepare_dataset(X_test, y_test, tokenizer)

        # Define training arguments (for evaluation)
        training_args = TrainingArguments(
            output_dir='./models',
            per_device_eval_batch_size=8,
            logging_dir='./logs',
            eval_strategy="no",
        )

        # Initialize trainer
        logger.info("Initializing trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics
        )

        # Evaluate
        logger.info("Evaluating model...")
        results = trainer.evaluate()
        logger.info(f"Test results: {results}")
        return results
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    evaluate_model()