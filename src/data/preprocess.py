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
lemmatizer = WordNetLemmatizer()

def preprocess_text(text, lang='en'):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#', '', text)  # Remove hashtags
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = emoji.demojize(text)  # Handle emojis
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words_en]
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

    print("Saving English splits...")
    try:
        pd.concat([X_train_en, y_train_en], axis=1).to_csv('data/train_en.csv', index=False)
        pd.concat([X_val_en, y_val_en], axis=1).to_csv('data/val_en.csv', index=False)
        pd.concat([X_test_en, y_test_en], axis=1).to_csv('data/test_en.csv', index=False)
        print("English splits saved successfully.")
    except Exception as e:
        print(f"Error saving English splits: {str(e)}")
        raise

    # Return only English splits
    return X_train_en, y_train_en, X_val_en, y_val_en, X_test_en, y_test_en

if __name__ == "__main__":
    load_and_preprocess()