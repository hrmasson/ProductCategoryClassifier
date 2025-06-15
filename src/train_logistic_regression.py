from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib
from src.preprocess import load_and_preprocess

def train_logistic_regression():
    df = load_and_preprocess("data/amazon_products.csv")
    X_train, X_test, y_train, y_test = train_test_split(df['product_text'], df['category'], test_size=0.2, random_state=42)
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')),
        ('clf', LogisticRegression(max_iter=1000))
    ])

    pipe.fit(X_train, y_train)
    joblib.dump(pipe, 'models/logistic_regression_model.pkl')

    return pipe

if __name__ == "__main__":
    train_logistic_regression()