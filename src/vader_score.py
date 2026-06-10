import argparse
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from src.preprocess import clean_text


def score_file(input_path: str, text_col: str, out_path: str):
    df = pd.read_csv(input_path)
    analyzer = SentimentIntensityAnalyzer()
    texts = df[text_col].fillna("")
    scores = []
    for t in texts:
        c = clean_text(t)
        s = analyzer.polarity_scores(c)
        scores.append(s)
    scores_df = pd.DataFrame(scores)
    res = pd.concat([df.reset_index(drop=True), scores_df], axis=1)
    res.to_csv(out_path, index=False)
    print(f"Wrote scored file to {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--text-col", default="review")
    p.add_argument("--out", default="vader_scored.csv")
    args = p.parse_args()
    score_file(args.input, args.text_col, args.out)


if __name__ == "__main__":
    main()
