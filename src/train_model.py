import os
import joblib
import numpy as np
import pandas as pd
from src.preprocess import preprocess_text
from src.utils import save_confusion_matrix
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

    # Split dataset with indices
    X_train, X_temp, y_train, y_temp, train_idx, temp_idx = train_test_split(X, y, df.index, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test, val_idx, test_idx = train_test_split(X_temp, y_temp, temp_idx, test_size=0.5, random_state=42)

    # Train Logistic Regression model
    param_grid = {"C": [1, 5, 10]}
    grid_search = GridSearchCV(LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42), 
                               param_grid, cv=5, scoring="f1")
    grid_search.fit(X_train, y_train)

    # Save trained model
    os.makedirs("./models", exist_ok=True)  # Ensure models directory exists
    model_path = "./models/trained_model.pkl"
    joblib.dump(grid_search.best_estimator_, model_path)
    print(f"\n Model saved to: {model_path}")

    # Ensure output directories exist
    os.makedirs("./docs", exist_ok=True)
    os.makedirs("./output", exist_ok=True)

    # File paths
    evaluation_file = "./docs/model_evaluation.txt"
    val_data_file = "./output/validation_data.csv"
    val_predictions_file = "./output/validation_predictions.csv"

    # Evaluate Model on Test Set
    y_test_pred = grid_search.best_estimator_.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_conf_matrix = confusion_matrix(y_test, y_test_pred)
    test_class_report = classification_report(y_test, y_test_pred)

    # Evaluate Model on Validation Set
    y_val_pred = grid_search.best_estimator_.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_conf_matrix = confusion_matrix(y_val, y_val_pred)
    val_class_report = classification_report(y_val, y_val_pred)

    # Save Validation Set to CSV using proper indices
    val_df = df.loc[val_idx, ["Cleaned_Description", "label"]]
    val_df.rename(columns={"label": "True_Label"}, inplace=True)
    val_df.to_csv(val_data_file, index=False)
    print(f"Validation dataset saved to: {val_data_file}")

    # Save Validation Predictions to CSV
    val_pred_df = val_df.copy()
    val_pred_df["Predicted_Label"] = y_val_pred
    val_pred_df.to_csv(val_predictions_file, index=False)
    print(f"Validation predictions saved to: {val_predictions_file}")

    # Prepare content for evaluation file
    evaluation_content = f"""
    --- Model Evaluation ---

    --- Test Set Evaluation ---
    Test Accuracy: {test_accuracy:.4f}
    Confusion Matrix:
    {test_conf_matrix}

    Classification Report:
    {test_class_report}

    --- Validation Set Evaluation ---
    Validation Accuracy: {val_accuracy:.4f}
    Confusion Matrix:
    {val_conf_matrix}

    Classification Report:
    {val_class_report}
    """

    # Print evaluation results to console
    print(evaluation_content)

    # Save evaluation results to file
    with open(evaluation_file, "w") as f:
        f.write(evaluation_content)
    print(f"\n Model evaluation results saved to: {evaluation_file}")

    # Save Confusion Matrices as PNG files
    save_confusion_matrix(test_conf_matrix, "test_confusion_matrix", "Test Set Confusion Matrix")
    save_confusion_matrix(val_conf_matrix, "validation_confusion_matrix", "Validation Set Confusion Matrix")

    return grid_search.best_estimator_