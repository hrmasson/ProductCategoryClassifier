# Product Category Classifier

This project predicts the category of an Amazon product based on its
name and description.

It uses two machine learning models:
- Logistic Regression + TF-IDF
- FastText (official implementation by Facebook)

The models are trained using this dataset:
 https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset

## Setup

Clone the repository or unzip the project archive.
Install required Python packages: `pip install -r requirements.txt`

# Train and evaluate the Models

`python src/train_fasttext.py`       # Train FastText
`python src/train_logreg.py`         # Train Logistic Regression + TF-IDF
`python src/evaluate.py`             # Evaluate both models

# Predict Categories

Start the interactive CLI app by running `python app.py` in the 'app' directory.
Enter product names or descriptions to get category predictions from both models.
To exit the app, type 'exit'.

# Folder Structure
project/
├── data/                # Contains the raw dataset (CSV)
├── models/              # Saved models (FastText and LogisticRegression)
├── src/                 # Training, preprocessing, and evaluation scripts
├── app/                 # Interactive CLI application
├── requirements.txt     # List of dependencies
└── README.md            # Project documentation
