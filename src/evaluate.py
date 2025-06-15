import joblib
import fasttext
from src.preprocess import load_and_preprocess
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def evaluate_models():
    df = load_and_preprocess("data/amazon_products.csv")
    X_train, X_test, y_train, y_test = train_test_split(df['product_text'], df['category'], test_size=0.2, random_state=42)

    print("Logistic Regression + TF-IDF")
    pipe = joblib.load("models/logistic_regression_model.pkl")
    y_pred_lr = pipe.predict(X_test)
    print(classification_report(y_test, y_pred_lr, zero_division=0))

    print("FastText")
    model = fasttext.load_model("models/fasttext_model.bin")
    y_pred_ft = [model.predict(text)[0][0].replace("__label__", "").replace("_", " ") for text in X_test]
    print(classification_report(y_test, y_pred_ft, zero_division=0))

if __name__ == "__main__":
    evaluate_models()
