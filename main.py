import argparse
import os
import pandas as pd
import joblib
from src.train_model import train
from src.predict import predict_injury

# Paths
MODEL_PATH = "./models/trained_model.pkl"
VECTORIZER_PATH = "./models/tfidf_vectorizer.pkl"

def main():
    parser = argparse.ArgumentParser(description="Case Injury Classifier - Train or Predict")
    parser.add_argument("mode", choices=["train", "predict"], help="Choose 'train' to train the model, 'predict' to classify text")
    parser.add_argument("--text", type=str, help="Text to classify (for single prediction)")
    parser.add_argument("--file", type=str, help="Path to an Excel file containing descriptions to classify")

    args = parser.parse_args()

    if args.mode == "train":
        print(" Training the model...")
        train()
        print(" Training complete. Model saved!")

    elif args.mode == "predict":
        if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
            print(" Error: Model not found. Train the model first using 'python main.py train'.")
            return

        if args.text:
            print(f"\n Predicting for input: {args.text}")
            prediction = predict_injury(args.text)
            print(f" Prediction: {prediction}")

        elif args.file:
            if not os.path.exists(args.file):
                print(f" Error: File '{args.file}' not found.")
                return

            print(f"\n Predicting for file: {args.file}")
            df = pd.read_excel(args.file)
            df["Prediction"] = df["Description"].apply(predict_injury)

            output_file = "./output/predictions.xlsx"
            df.to_excel(output_file, index=False)
            print(f" Predictions saved to {output_file}")

        else:
            print(" Error: Provide either --text or --file for predictions.")

if __name__ == "__main__":
    main()