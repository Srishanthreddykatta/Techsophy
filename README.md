#  Smart Email Classifier System

An intelligent multi-label email classification system that automatically categorizes emails by **spam/ham** status and **priority level** (irrelevant, high, medium, low). Built with modular Python, this system demonstrates NLP preprocessing, multi-model classification, and a FastAPI-powered prediction API.

---

##  Features

-  **Multi-Task Classification**

  - Spam vs Ham
  - Priority Levels (0: Irrelevant, 1: High, 2: Medium, 3: Low)

-  **Machine Learning Models**

  - Naive Bayes and SVM support
  - TF-IDF feature extraction
  - Handles class imbalance effectively

-  **Text Preprocessing**

  - Tokenization, stopword removal, lemmatization
  - Noise removal (HTML, special characters, etc.)

-  **Modular Design**

  - Clean separation of preprocessing, vectorization, model training, and inference
  - Easy to extend for department-wise classification or other tasks

-  **FastAPI Server**

  - REST API endpoint to classify emails in real time
  - JSON output with predicted labels and confidence scores

---

## Project Structure

```
SmartEmailClassifier/
├── data/                    # CSV datasets for each classification task
├── models/                  # Saved models and vectorizers (SVM, NB)
│   ├── svm/
│   │   ├── spam/
│   │   └── priority/
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineeering.py
│   ├── select_model.py
│   ├── predictor.py
├── fastapi/
│   └── server.py            # FastAPI endpoint for multi-task predictions
├── trainer.py               # Script to train models for each task
├── test_email.py            # Simple CLI test script for local predictions
├── requirements.txt
└── README.md
```

---

## Example Output

```json
{
  "spam": {
    "label": "ham",
    "confidence": 0.97
  },
  "priority": {
    "label": "high",
    "confidence": 0.89
  }
}
```

---

## Installation

```bash
git clone https://github.com/Srishanthreddykatta/Techsophy
cd Techsophy
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

---

## Training

Place your labeled datasets in `data/` and run:

```bash
python trainer.py
```

This will train and save models for spam/ham and priority tasks.

---

## Testing Locally

```bash
python main.py
```
or

```bash
python test.py
```

It will prompt for an email input and return both predictions.

---

## Running API

```bash
uvicorn server:app --reload
```

POST request:

```json
{
  "email": "Your email text here"
}
```

---

## Future Scope

- Department-wise email routing
- BERT-based models for higher accuracy
- Dashboard for visualizing results

---

## Acknowledgements

Built using Python, Scikit-learn, NLTK, and FastAPI.

