from sklearn.feature_extraction.text import TfidfVectorizer

def extract_features(corpus):
    vectorizer = TfidfVectorizer(max_features=3000)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features
