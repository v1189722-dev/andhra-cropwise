from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "processed" / "crop_recommendation_clean.csv"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
]
TARGET = "label"


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. Run src/prepare_data.py first."
        )

    data = pd.read_csv(DATA_PATH)
    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(
        y_test,
        predictions,
        zero_division=0,
    )

    model_path = MODEL_DIR / "crop_recommender.joblib"
    report_path = MODEL_DIR / "crop_recommender_report.txt"
    metadata_path = MODEL_DIR / "crop_recommender_metadata.json"

    joblib.dump(
        {"model": model, "features": FEATURES},
        model_path,
    )
    report_path.write_text(report, encoding="utf-8")
    metadata_path.write_text(
        json.dumps(
            {
                "accuracy_on_holdout": accuracy,
                "rows": int(len(data)),
                "crop_labels": sorted(y.unique().tolist()),
                "features": FEATURES,
                "note": (
                    "The crop recommendation dataset has no district or season "
                    "columns. This model uses soil and weather inputs only."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Training rows: {len(X_train):,}")
    print(f"Held-out rows: {len(X_test):,}")
    print(f"Holdout accuracy: {accuracy:.1%}")
    print(f"Model saved to: {model_path}")
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()