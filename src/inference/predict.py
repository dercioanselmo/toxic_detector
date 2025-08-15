import torch
from transformers import BertTokenizer, BertForSequenceClassification

def predict(text):
    try:
        # Load tokenizer and model
        tokenizer = BertTokenizer.from_pretrained('models/bert-toxic-detector')
        model = BertForSequenceClassification.from_pretrained('models/bert-toxic-detector')
        
        # Tokenize input
        encodings = tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors='pt')
        
        # Run inference
        with torch.no_grad():
            outputs = model(**encodings)
        probs = torch.sigmoid(outputs.logits).numpy()[0]
        
        # Map probabilities to labels
        labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate', 'neutral']
        return {label: float(prob) for label, prob in zip(labels, probs)}
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None

if __name__ == "__main__":
    test_text = "This is a test comment"
    result = predict(test_text)
    print(result)