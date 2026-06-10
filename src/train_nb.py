import argparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from preprocess import clean_text


def train_and_save(input_path: str, text_col: str, label_col: str, model_out: str):
    df = pd.read_csv(input_path)
    df = df[[text_col, label_col]].dropna()
    texts = df[text_col].astype(str).apply(clean_text).tolist()
    labels = df[label_col].astype(str).tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    vect = TfidfVectorizer(max_features=20000)
    X_train_t = vect.fit_transform(X_train)
    X_test_t = vect.transform(X_test)

    clf = MultinomialNB()
    clf.fit(X_train_t, y_train)

    preds = clf.predict(X_test_t)
    print(classification_report(y_test, preds))

    joblib.dump({"vectorizer": vect, "model": clf}, model_out)
    print(f"Saved model to {model_out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--text-col", default="review")
    p.add_argument("--label-col", default="label")
    p.add_argument("--model-out", default="model.joblib")
    args = p.parse_args()
    train_and_save(args.input, args.text_col, args.label_col, args.model_out)


if __name__ == "__main__":
    main()
