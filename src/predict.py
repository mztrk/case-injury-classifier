import os
import sys
import joblib
import pandas as pd
from src.preprocess import preprocess_text  # Ensure preprocessing is applied
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model and TF-IDF vectorizer
MODEL_PATH = "./models/trained_model.pkl"
VECTORIZER_PATH = "./models/tfidf_vectorizer.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    raise FileNotFoundError("Model or TF-IDF vectorizer not found! Train the model first.")

# Load the saved model
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

def predict_injury(text):
    """
    Predicts whether the given text describes an injury-related incident.
    """
    # Preprocess text (translation + cleaning)
    processed_text = preprocess_text(pd.DataFrame({"Description": [text]}))["Cleaned_Description"].iloc[0]

    # Convert text to TF-IDF features
    text_vectorized = vectorizer.transform([processed_text])

    # Make prediction
    prediction = model.predict(text_vectorized)[0]
    return "Injury" if prediction == 1 else "Non-Injury"

if __name__ == "__main__":
    # Accept input from the command line
    if len(sys.argv) < 2:
        print("Usage: python scripts/predict.py 'Your text here'")
        sys.exit(1)

    input_text = sys.argv[1]
    result = predict_injury(input_text)
    print(f"\nPrediction: {result}")