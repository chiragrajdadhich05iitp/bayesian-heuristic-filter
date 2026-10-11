import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from preprocess import preprocess

def train_model():
    # Load dataset
    data = pd.read_csv("spam.csv", encoding='latin-1')
    data = data[['Category', 'Message']]
    data.columns = ['label', 'text']

    # Convert labels
    data['label'] = data['label'].map({'ham': 0, 'spam': 1})

    # Clean text
    data['clean'] = data['text'].apply(preprocess)

    # Vectorization
    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=7000)
    X = vectorizer.fit_transform(data['clean'])
    y = data['label']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train
    model = MultinomialNB(alpha=0.7)
    model.fit(X_train, y_train)

    return model, vectorizer