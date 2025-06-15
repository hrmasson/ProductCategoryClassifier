import fasttext
from src.preprocess import load_and_preprocess

def save_fasttext_input(df, path='data/fasttext_input.txt'):
    with open(path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            label = "__label__" + row['category'].replace(" ", "_")
            f.write(f"{label} {row['product_text']}\n")

def train_fasttext():
    df = load_and_preprocess("data/amazon_products.csv")
    save_fasttext_input(df)
    model = fasttext.train_supervised('data/fasttext_input.txt', epoch=25, lr=1.0, wordNgrams=2)
    model.save_model("models/fasttext_model.bin")
    return model

if __name__ == "__main__":
    train_fasttext()
