import joblib
from src.data_preprocessing import preprocess_text

# Label name maps
LABEL_NAMES = {
    "spam": {
        0: "ham",
        1: "spam"
    },
    "priority": {
        0: "irrelevant",
        1: "high",
        2: "medium",
        3: "low"
    }
}

# Reverse mapping to support string-based label prediction
REVERSE_LABELS = {
    "spam": {v: k for k, v in LABEL_NAMES["spam"].items()},
    "priority": {v: k for k, v in LABEL_NAMES["priority"].items()}
}


def load_model_and_vectorizer(model_type, task):
    model_path = f"models/{model_type}/{task}/classifier.pkl"
    vectorizer_path = f"models/{model_type}/{task}/vectorizer.pkl"

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer


def predict_email_multi(email_text, model_type="svm"):
    cleaned = preprocess_text(email_text)
    results = {}

    for task in ["spam", "priority"]:
        model, vectorizer = load_model_and_vectorizer(model_type, task)
        X = vectorizer.transform([cleaned])

        raw_pred = model.predict(X)[0]

        # If model returns a string (like 'ham'), use as is
        if isinstance(raw_pred, str):
            label = raw_pred
        else:
            label = LABEL_NAMES[task].get(raw_pred, "unknown")

        # Confidence
        if hasattr(model, "predict_proba"):
            conf = max(model.predict_proba(X)[0])
        else:
            conf = 1.0

        results[task] = {
            "label": label,
            "confidence": round(conf, 4)
        }

    return results
