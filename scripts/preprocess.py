import re
import string
import time
import pandas as pd
import nltk
from deep_translator import GoogleTranslator
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Initialize NLP tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# --- Text Translation ---
def translate_to_english(text):
    """Translates text to English using Google Translator."""
    try:
        text = str(text).strip()
        return GoogleTranslator(source="auto", target="en").translate(text)
    except Exception:
        return text  # If translation fails, return the original text

# --- Text Cleaning ---
def clean_text(text):
    """Cleans and preprocesses text."""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    tokens = word_tokenize(text)  # Tokenization
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatization
    return " ".join(tokens)  # Convert tokens back to string

# --- Preprocessing Pipeline ---
def preprocess_text(df):
    """Applies translation and text preprocessing."""
    print(" Translating text to English...")
    df["Translated_Description"] = df["Description"].apply(translate_to_english)

    print(" Cleaning text...")
    df["Cleaned_Description"] = df["Translated_Description"].apply(clean_text)

    return df