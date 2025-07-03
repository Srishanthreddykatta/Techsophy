from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

def train_model(X, y, model_type="nb"):
    if model_type == "nb":
        model = MultinomialNB()
    elif model_type == "svm":
        model = SVC(probability=True)
    else:
        raise ValueError("Invalid model type")

    model.fit(X, y)
    return model
