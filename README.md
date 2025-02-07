# Injury Incident Classifier

For the deliverables please check Safety_Incident_EDA_Model_Mustafa.ipynb.

## Project Overview
This project aims to build a machine learning model that predicts whether an airline safety incident report describes an injury-related event (1) or not (0). The model is trained on textual descriptions of safety incidents.

## Features
- **Data Preprocessing**: Text cleaning, feature extraction (TF-IDF), optional language translation.
- **Model Training**: Logistic Regression, Random Forest, and SVM classifiers.
- **Evaluation**: Accuracy, confusion matrix, classification report.
- **Visualization**: Confusion matrix plots for model performance understanding.

---

## Project Structure
```
case-injury-classifier/
│── data/                                     # Dataset files
│   ├── safety_incident_reports.xlsx
│── models/                                   # Saved models
│   ├── trained_model.pkl
│   ├── tfidf_vectorizer.pkl
│── docs/                                     # Model evaluation and reports
│   ├── model_evaluation.docx
│   ├── predictions.xlsx
│── src/                                      # Source scripts
│   ├── __init__.py
│   ├── train_model.py
│   ├── predict.py
│   ├── preprocess.py
│── main.py                                   # Entry point for training & prediction
│── requirements.txt                          # Required dependencies
│── README.md                                 # Project documentation
│── safety_incident_reports.xlsx              # Dataset to work on
│── Safety_Incident_EDA_Model_Mustafa.ipynb   # Deliverabels are here. Please check it. 
```

---

## Setup Guide

###  1. Clone the Repository
```sh
git clone https://github.com/your-username/case-injury-classifier.git
cd case-injury-classifier
```

#### 2. Create a Virtual Environment
```sh
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate   # On Windows
```

### 3. Install Dependencies
Ensure all required libraries are installed using the following command:
```sh
pip install -r requirements.txt
```

## Usage
### Train the Model

Note: **You do not have to train the model.** 

.pkl files already in the ./models folder.

To train the model, run:
```sh
python main.py train
```

This will:
- Load and preprocess the dataset.
- Train a Logistic Regression model using TF-IDF features.
- Save the trained model and vectorizer inside models/.


### Make Predictions

Predict a Single Description
```sh
python main.py predict --text "A worker fell and broke his leg on the tarmac."
```
Predict for a File
```sh
python main.py predict --file ./data/safety_incident_reports.xlsx
```
This will generate predictions for all descriptions in the file and save them to ./docs/predictions.xlsx.

---

## Implementation Details

### 1. Data Preprocessing
The preprocessing pipeline includes:
- **Translation**: Automatically translates text to English.
- **Text Cleaning**: Removes punctuation, numbers, and stopwords.
- **Lemmatization**: Converts words to their root form.

### 2. Feature Engineering
- **TF-IDF Vectorization**: Converts text into numerical features.
- **N-grams**: Uses both unigrams and bigrams to preserve context.

### 3. Model Training
- **Algorithm**: Logistic Regression with class balancing.
- **Hyperparameter Tuning**: Uses GridSearchCV to optimize regularization strength.
- **Evaluation Metrics**:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - Confusion Matrix

### 4. Model Evaluation
The evaluation results are saved in `./docs/model_evaluation.docx` and include:
- **Classification report**
- **Confusion matrix**
- **Test accuracy and validation accuracy**

---

Troubleshooting

1. Import Errors

If you face ModuleNotFoundError, ensure __init__.py exists in the scripts/ directory:
```sh
touch src/__init__.py
```
2. Missing Models

If prediction fails due to missing models, train the model first:
```sh
python main.py train
```
3. Virtual Environment Not Found

If pip installs fail, ensure you activated the virtual environment:

```sh
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows
```


---

## Author
Created by Mustafa Ozturk

For questions or improvements, feel free to contribute or reach out!

