import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer
import string
import emoji
import re
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

stop_words_en = set(stopwords.words('english'))
stop_words_ar = set(stopwords.words('arabic'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def preprocess_text(text, lang='en'):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#', '', text)  # Remove hashtags
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = emoji.demojize(text)  # Handle emojis
    words = text.split()
    if lang == 'en':
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words_en]
    else:
        words = [stemmer.stem(word) for word in words if word not in stop_words_ar]  # Simple stemming for Arabic
    return ' '.join(words)

def load_and_preprocess():
    print("Loading English dataset...")
    try:
        df_en = pd.read_csv('data/english/train.csv')
        print(f"English dataset shape: {df_en.shape}")
        df_en['comment_text'] = df_en['comment_text'].apply(preprocess_text, lang='en')
        labels_en = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        df_en['neutral'] = (df_en[labels_en].sum(axis=1) == 0).astype(int)
        X_en = df_en['comment_text']
        y_en = df_en[labels_en + ['neutral']]
        X_train_en, X_temp_en, y_train_en, y_temp_en = train_test_split(X_en, y_en, test_size=0.2, random_state=42)
        X_val_en, X_test_en, y_val_en, y_test_en = train_test_split(X_temp_en, y_temp_en, test_size=0.5, random_state=42)
        print(f"English splits: train={X_train_en.shape}, val={X_val_en.shape}, test={X_test_en.shape}")
    except Exception as e:
        print(f"Error processing English dataset: {str(e)}")
        raise

    print("Loading Arabic dataset...")
    try:
        df_ar = pd.read_csv('data/arabic/l_hsab.csv')
        print(f"Arabic dataset shape: {df_ar.shape}")
        print(f"Arabic columns: {df_ar.columns.tolist()}")
        df_ar['Tweet'] = df_ar['Tweet'].apply(preprocess_text, lang='ar')
        label_mapping = {
            'normal': [0, 0, 0, 0, 0, 0, 1],  # neutral
            'abusive': [1, 0, 1, 0, 1, 0, 0],  # toxic, obscene, insult
            'hate': [1, 1, 0, 1, 0, 1, 0]  # toxic, severe_toxic, threat, identity_hate
        }
        y_ar = pd.DataFrame([label_mapping[label.lower()] for label in df_ar['Class']], columns=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate', 'neutral'])
        X_ar = df_ar['Tweet']
        X_train_ar, X_temp_ar, y_train_ar, y_temp_ar = train_test_split(X_ar, y_ar, test_size=0.2, random_state=42)
        X_val_ar, X_test_ar, y_val_ar, y_test_ar = train_test_split(X_temp_ar, y_temp_ar, test_size=0.5, random_state=42)
        print(f"Arabic splits: train={X_train_ar.shape}, val={X_val_ar.shape}, test={X_test_ar.shape}")
    except Exception as e:
        print(f"Error processing Arabic dataset: {str(e)}")
        raise

    print("Saving splits...")
    try:
        pd.concat([X_train_en, y_train_en], axis=1).to_csv('data/train_en.csv', index=False)
        pd.concat([X_val_en, y_val_en], axis=1).to_csv('data/val_en.csv', index=False)
        pd.concat([X_test_en, y_test_en], axis=1).to_csv('data/test_en.csv', index=False)
        pd.concat([X_train_ar, y_train_ar], axis=1).to_csv('data/train_ar.csv', index=False)
        pd.concat([X_val_ar, y_val_ar], axis=1).to_csv('data/val_ar.csv', index=False)
        pd.concat([X_test_ar, y_test_ar], axis=1).to_csv('data/test_ar.csv', index=False)
        print("Splits saved successfully.")
    except Exception as e:
        print(f"Error saving splits: {str(e)}")
        raise

    return X_train_en, y_train_en, X_val_en, y_val_en, X_test_en, y_test_en, X_train_ar, y_train_ar, X_val_ar, y_val_ar, X_test_ar, y_test_ar

if __name__ == "__main__":
    load_and_preprocess()