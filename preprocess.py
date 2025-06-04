import pandas as pd
import re
from sklearn.model_selection import train_test_split

def load_and_clean_data(path):
    df = pd.read_csv(path)

    # Filter: we'll do binary classification for simplicity (positive vs negative)
    df = df[df['stars'] != 3]  # drop neutral
    df['label'] = df['stars'].apply(lambda x: 1 if x >= 4 else 0)  # 1=positive, 0=negative

    # Simple text cleaning
    df['text_clean'] = df['text'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', str(x).lower()))

    return train_test_split(df['text_clean'], df['label'], test_size=0.2, random_state=42)
