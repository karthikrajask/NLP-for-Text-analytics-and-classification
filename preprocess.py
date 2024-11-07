import pandas as pd
import re
from nltk.corpus import stopwords

data = pd.read_csv('train.csv')
import nltk
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def advanced_preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'http[s]?://\S+|www\.\S+', '', text)
        text = re.sub(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b', '', text)
        text = re.sub(r'\+?\d[\d -]{8,12}\d', '', text)  
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = [word for word in text.split() if len(word) > 1 and word not in stop_words]
        return ' '.join(tokens)
    else:
        return ''  # Return an empty string for non-text entries

data['processed_crime_info'] = data['crimeaditionalinfo'].apply(advanced_preprocess_text)

print(data[['crimeaditionalinfo', 'processed_crime_info']].head())
data.to_csv('D:/Downloads/english.csv', index=False)
