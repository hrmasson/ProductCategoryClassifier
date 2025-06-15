import pandas as pd
import re


def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)

    return text.strip()

def load_and_preprocess(file_path):
    df = pd.read_csv(file_path)
    product_text = df['product_name'].astype(str) + ' ' + df['about_product'].astype(str)
    df['product_text'] = product_text.apply(clean_text)
    df['category'] = df['category'].str.split('|').str[0]
    df = df[['product_text', 'category']].dropna()

    return df
    