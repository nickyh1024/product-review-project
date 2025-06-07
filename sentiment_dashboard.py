# sentiment_dashboard.py
import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load data
df = pd.read_csv("yelp_sample.csv")
df = df[df['stars'] != 3]
df['label'] = df['stars'].apply(lambda x: 1 if x >= 4 else 0)

# Load model and vectorizer (we'll save them next)
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# UI
st.title("Yelp Sentiment Analysis Dashboard")
st.write("Predict and explore sentiment of product reviews")

# Sentiment breakdown
st.subheader("Sentiment Distribution")
st.bar_chart(df['label'].value_counts())

# Review explorer
st.subheader("Try It Yourself")
review_text = st.text_area("Enter a review:")
if review_text:
    vectorized = vectorizer.transform([review_text])
    pred = model.predict(vectorized)[0]
    label = "Positive 👍" if pred == 1 else "Negative 👎"
    st.write(f"**Prediction:** {label}")
