import os
import zipfile
import requests
from io import BytesIO
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

# English dataset
api = KaggleApi()
api.authenticate()
os.makedirs('data/english', exist_ok=True)
api.competition_download_files('jigsaw-toxic-comment-classification-challenge', path='data/english')
with zipfile.ZipFile('data/english/jigsaw-toxic-comment-classification-challenge.zip', 'r') as zip_ref:
    zip_ref.extractall('data/english')
os.remove('data/english/jigsaw-toxic-comment-classification-challenge.zip')  # Clean up
# Unzip inner zips
for file in os.listdir('data/english'):
    if file.endswith('.zip'):
        with zipfile.ZipFile(f'data/english/{file}', 'r') as zip_ref:
            zip_ref.extractall('data/english')
        os.remove(f'data/english/{file}')

# Arabic dataset from GitHub
os.makedirs('data/arabic', exist_ok=True)
url = 'https://raw.githubusercontent.com/Hala-Mulki/L-HSAB-First-Arabic-Levantine-HateSpeech-Dataset/master/L-HSAB.csv'  # Assume this URL, adjust if different
response = requests.get(url)
if response.status_code == 200:
    df = pd.read_csv(BytesIO(response.content))
    df.to_csv('data/arabic/l_hsab.csv', index=False)
else:
    print("Failed to download Arabic dataset. Please download manually from GitHub.")