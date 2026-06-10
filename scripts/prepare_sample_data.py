import csv
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "sample_netflix_reviews.csv"

def ensure_sample():
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        rows = [
            ("I loved this series, the acting was superb and plot addictive.", "positive"),
            ("Terrible season; plot holes and poor pacing.", "negative"),
            ("Decent show, some episodes were slow but overall enjoyable.", "positive"),
            ("Not my cup of tea, found it boring.", "negative"),
            ("An outstanding documentary — informative and moving.", "positive"),
            ("Soundtrack was great but storyline lacked depth.", "neutral"),
        ]
        with open(DATA_PATH, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["review", "label"]) 
            w.writerows(rows)
        print(f"Wrote sample data to {DATA_PATH}")
    else:
        print(f"Sample data already exists at {DATA_PATH}")

if __name__ == "__main__":
    ensure_sample()
