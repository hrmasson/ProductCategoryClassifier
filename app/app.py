import fasttext
import joblib

ft_model = fasttext.load_model("../models/fasttext_model.bin")
lg_model = joblib.load("../models/logistic_regression_model.pkl")

def predict_fasttext(text):
    label = ft_model.predict(text)[0][0]
    return label.replace("__label__", "").replace("_", " ")

def predict_logreg(text):
    return lg_model.predict([text])[0]

print("Amazon Category Classifier")
print("Enter product description (or ‘exit’ to exit)):")


while True:
    user_input = input("Enter product description: ")
    
    if user_input.lower() in ['exit']:
        break
    cleaned_input = user_input.lower()

    pred_ft = predict_fasttext(cleaned_input)
    pred_lr = predict_logreg(cleaned_input)

    print(f"\nFastText predict:         {pred_ft}")
    print(f"Logistic Regression predict: {pred_lr}")
