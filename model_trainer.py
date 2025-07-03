import pandas as pd
from src.data_preprocessing import preprocess_text
from src.feature_engineeering import extract_features
from src.model_selector import train_model
from src.predictor import predict_email
import joblib
import os
import pandas as pd

splits = {'train': 'train.csv', 'validation': 'val.csv', 'test': 'test.csv'}
df = pd.read_csv("hf://datasets/vedaantp/email-priority/" + splits["train"])
# Configuration: task_name -> (CSV file, label column)
tasks = {
    "spam": ("data/spam-ham.csv", "label"),
    "priority": ("data/priority.csv", "label")
}

# Model type to use for all tasks: "nb" or "svm"
model_type = "nb"

# Train and save model/vectorizer for each task
for task_name, (csv_file, label_col) in tasks.items():
    print(f"\n Processing task: {task_name.upper()}")

    # Load and preprocess
    df = pd.read_csv(csv_file)
    df['cleaned'] = df['text'].apply(preprocess_text)

    # Feature extraction (vectorizer is task-specific)
    vectorizer, X = extract_features(df['cleaned'])
    y = df[label_col]

    # Train
    model = train_model(X, y, model_type=model_type)

    # Save model & vectorizer
    model_dir = f"models/{model_type}/{task_name}"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, f"{model_dir}/classifier.pkl")
    joblib.dump(vectorizer, f"{model_dir}/vectorizer.pkl")

    print(f" Model saved at: {model_dir}")

# Prediction for a new email
email = "VPN issue reported to IT, needs urgent attention."

print(f"\n Predicting for: '{email}'\n")

for task_name in tasks:
    model_path = f"models/{model_type}/{task_name}/classifier.pkl"
    vectorizer_path = f"models/{model_type}/{task_name}/vectorizer.pkl"
    label, confidence = predict_email(email, model_path, vectorizer_path)
    print(f"{task_name.upper()} → {label} (Confidence: {confidence:.2f})")
