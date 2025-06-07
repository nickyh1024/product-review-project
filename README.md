# product-review-project

This project uses natural language processing (NLP) and machine learning to analyze the sentiment of Yelp product reviews.

## Features
- Preprocesses Yelp review text (lowercase, clean punctuation)
- Creates binary sentiment labels: positive (4–5 stars), negative (1–2 stars)
- Vectorizes text using TF-IDF
- Trains a Logistic Regression classifier
- Evaluates model with classification report and confusion matrix

## Current Results (TF-IDF + Logistic Regression)
- **Accuracy**: 92%
- **F1-score (Positive)**: 0.95
- **F1-score (Negative)**: 0.79

## 🔜 Next Steps
- Try BERT-based transformer models
- Balance the dataset or tune class weights
- Deploy model insights in an interactive dashboard

## Dataset
- Subset from [Yelp Dataset](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset)
- Only `yelp_sample.csv` (10K reviews) included to avoid large files

---

## 🛠 Setup
```bash
pip install -r requirements.txt
python train_model.py
