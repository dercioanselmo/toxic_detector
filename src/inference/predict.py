import joblib
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from langdetect import detect
from src.data.preprocess import preprocess_text
from src.features.vectorize import tfidf_vectorize  # But for inference, load fitted tfidf
import pickle

# Load models
lr_en = joblib.load('models/lr_en.joblib')
# Similarly for others
bert_en = BertForSequenceClassification.from_pretrained('models/bert_en')
tokenizer_en = BertTokenizer.from_pretrained('models/bert_en')
bert_ar = BertForSequenceClassification.from_pretrained('models/bert_ar')
tokenizer_ar = BertTokenizer.from_pretrained('models/bert_ar')

# Load fitted tfidf if needed

def predict(text, model_type='bert'):
    lang = detect(text)
    processed = preprocess_text(text, lang)
    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate', 'neutral']
    if model_type != 'bert':
        # Assume tfidf loaded as tfidf_en, tfidf_ar
        vec = tfidf_en.transform([processed]) if lang == 'en' else tfidf_ar.transform([processed])
        model = lr_en if lang == 'en' else lr_ar  # Assume lr_ar trained
        probs = model.predict_proba(vec)[0]
        preds = (probs > 0.5).astype(int)
    else:
        if lang == 'en':
            inputs = tokenizer_en(processed, return_tensors='pt', truncation=True, padding=True)
            outputs = bert_en(**inputs)
        else:
            inputs = tokenizer_ar(processed, return_tensors='pt', truncation=True, padding=True)
            outputs = bert_ar(**inputs)
        probs = torch.sigmoid(outputs.logits).detach().numpy()[0]
        preds = (probs > 0.5).astype(int)
    result = {label: {'pred': pred, 'confidence': prob} for label, pred, prob in zip(labels, preds, probs)}
    return result

# Example
print(predict("This is a bad comment"))