import joblib
import numpy as np
import pandas as pd
from scripts.preprocess import preprocess_text
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def train():
    dataset_path = "./data/safety_incident_reports.xlsx"
    df = pd.read_excel(dataset_path)

    # Preprocess text
    df = preprocess_text(df)

    # Convert text to numerical features using TF-IDF
    vectorizer = TfidfVectorizer(max_features=6000, stop_words="english", ngram_range=(1,2), sublinear_tf=True)
    X = vectorizer.fit_transform(df["Cleaned_Description"])
    y = df["label"]

    # Save TF-IDF model
    joblib.dump(vectorizer, "./models/tfidf_vectorizer.pkl")

    # Split dataset
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Train Logistic Regression model
    param_grid = {"C": [1, 5, 10]}
    grid_search = GridSearchCV(LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42), 
                               param_grid, cv=5, scoring="f1")
    grid_search.fit(X_train, y_train)

    # Save model
    joblib.dump(grid_search.best_estimator_, "./models/trained_model.pkl")

    # Evaluate Model
    y_pred = grid_search.best_estimator_.predict(X_test)
    print("\n--- Model Evaluation ---")
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return grid_search.best_estimator_