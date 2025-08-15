import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer
import string
import emoji
import re
from sklearn.model_selection import train_test_split
#from imblearn.over_sampling import RandomOverSampler
from imbalanced_learn.over_sampling import RandomOverSampler


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
    # English
    df_en = pd.read_csv('data/english/train.csv')
    df_en['comment_text'] = df_en['comment_text'].apply(preprocess_text, lang='en')
    labels_en = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    df_en['neutral'] = (df_en[labels_en].sum(axis=1) == 0).astype(int)
    X_en = df_en['comment_text']
    y_en = df_en[labels_en + ['neutral']]
    X_train_en, X_temp_en, y_train_en, y_temp_en = train_test_split(X_en, y_en, test_size=0.2, random_state=42)
    X_val_en, X_test_en, y_val_en, y_test_en = train_test_split(X_temp_en, y_temp_en, test_size=0.5, random_state=42)

    # Arabic
    df_ar = pd.read_csv('data/arabic/l_hsab.csv')
    df_ar['Text'] = df_ar['Text'].apply(preprocess_text, lang='ar')  # Assume column 'Text'
    # Assume column 'Label' with 'Normal', 'Abusive', 'Hate'
    label_mapping = {
        'Normal': [0,0,0,0,0,0,1],  # neutral
        'Abusive': [1,0,1,0,1,0,0],  # toxic, obscene, insult
        'Hate': [1,1,0,1,0,1,0]  # toxic, severe_toxic, threat, identity_hate
    }
    y_ar = pd.DataFrame([label_mapping[label] for label in df_ar['Label']], columns=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate', 'neutral'])
    X_ar = df_ar['Text']
    X_train_ar, X_temp_ar, y_train_ar, y_temp_ar = train_test_split(X_ar, y_ar, test_size=0.2, random_state=42)
    X_val_ar, X_test_ar, y_val_ar, y_test_ar = train_test_split(X_temp_ar, y_temp_ar, test_size=0.5, random_state=42)

    # Save splits
    pd.concat([X_train_en, y_train_en], axis=1).to_csv('data/train_en.csv', index=False)
    pd.concat([X_val_en, y_val_en], axis=1).to_csv('data/val_en.csv', index=False)
    pd.concat([X_test_en, y_test_en], axis=1).to_csv('data/test_en.csv', index=False)
    pd.concat([X_train_ar, y_train_ar], axis=1).to_csv('data/train_ar.csv', index=False)
    pd.concat([X_val_ar, y_val_ar], axis=1).to_csv('data/val_ar.csv', index=False)
    pd.concat([X_test_ar, y_test_ar], axis=1).to_csv('data/test_ar.csv', index=False)

    return X_train_en, y_train_en, X_val_en, y_val_en, X_test_en, y_test_en, X_train_ar, y_train_ar, X_val_ar, y_val_ar, X_test_ar, y_test_ar