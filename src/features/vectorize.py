from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from transformers import BertTokenizer, BertModel
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
bert_model = BertModel.from_pretrained('bert-base-multilingual-uncased').to(device)

def tfidf_vectorize(X_train, X_val, X_test):
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2), analyzer='word')
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)
    X_test_tfidf = tfidf.transform(X_test)
    tfidf_char = TfidfVectorizer(max_features=5000, ngram_range=(3,5), analyzer='char')
    X_train_char = tfidf_char.fit_transform(X_train)
    X_val_char = tfidf_char.transform(X_val)
    X_test_char = tfidf_char.transform(X_test)
    return np.hstack([X_train_tfidf.toarray(), X_train_char.toarray()]), np.hstack([X_val_tfidf.toarray(), X_val_char.toarray()]), np.hstack([X_test_tfidf.toarray(), X_test_char.toarray()])

def get_bert_embeddings(X, batch_size=32):
    embeddings = []
    for i in range(0, len(X), batch_size):
        batch = X[i:i+batch_size]
        inputs = tokenizer(batch.tolist(), padding=True, truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        embeddings.append(outputs.pooler_output.cpu().numpy())
    return np.vstack(embeddings)

# For GloVe/FastText, download and average word vectors (code omitted for brevity, similar to BERT)