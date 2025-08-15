import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import OneVsRestClassifier
from xgboost import XGBClassifier
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import ParameterSampler
from optuna import create_study
from joblib import dump
from src.features.vectorize import tfidf_vectorize, get_bert_embeddings
from imbalancedlearn.over_sampling import RandomOverSampler
import numpy as np
from sklearn.metrics import f1_score

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(texts.tolist(), truncation=True, padding=True, max_length=128)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels.iloc[idx].values, dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)

def train_traditional(model_type, X_train, y_train, X_val, y_val):
    ros = RandomOverSampler(random_state=42)
    X_train_res, y_train_res = ros.fit_resample(X_train, y_train)
    if model_type == 'lr':
        model = OneVsRestClassifier(LogisticRegression())
    elif model_type == 'rf':
        model = OneVsRestClassifier(RandomForestClassifier())
    elif model_type == 'xgb':
        model = OneVsRestClassifier(XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
    model.fit(X_train_res, y_train_res)
    return model

def optimize_hyperparams(model_type, X_train, y_train, X_val, y_val):
    def objective(trial):
        if model_type == 'xgb':
            params = {'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3), 'max_depth': trial.suggest_int('max_depth', 3, 10)}
            model = OneVsRestClassifier(XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss'))
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            return f1_score(y_val, preds, average='macro')
    study = create_study(direction='maximize')
    study.optimize(objective, n_trials=10)
    return study.best_params

def train_bert(X_train, y_train, X_val, y_val, lang):
    model_name = 'bert-base-uncased' if lang == 'en' else 'aubmindlab/bert-base-arabertv02'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=7, problem_type='multi_label_classification')
    train_dataset = TextDataset(X_train, y_train, tokenizer)
    val_dataset = TextDataset(X_val, y_val, tokenizer)
    training_args = TrainingArguments(output_dir='./results', num_train_epochs=3, per_device_train_batch_size=16, per_device_eval_batch_size=16, warmup_steps=500, weight_decay=0.01, logging_dir='./logs', load_best_model_at_end=True)
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset)
    trainer.train()
    model.save_pretrained(f'models/bert_{lang}')
    tokenizer.save_pretrained(f'models/bert_{lang}')
    return model, tokenizer

# Example usage
# Load from preprocess
X_train_en, y_train_en, ... = load_and_preprocess()  # From preprocess
X_train_tfidf_en, X_val_tfidf_en, X_test_tfidf_en = tfidf_vectorize(X_train_en, X_val_en, X_test_en)
# Train LR for English
model_lr_en = train_traditional('lr', X_train_tfidf_en, y_train_en, X_val_tfidf_en, y_val_en)
dump(model_lr_en, 'models/lr_en.joblib')
# Similarly for other models and Arabic
# For BERT
bert_en, tokenizer_en = train_bert(X_train_en, y_train_en, X_val_en, y_val_en, 'en')
# Compare performance in evaluation