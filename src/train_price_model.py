from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "andhra_pradesh_prices_available.csv"
)
MODEL_PATH = PROJECT_DIR / "models" / "price_predictor.joblib"
REPORT_PATH = PROJECT_DIR / "models" / "price_predictor_report.txt"

CATEGORICAL_FEATURES = ["District", "Commodity"]
NUMERIC_FEATURES = ["Month", "DayOfYear"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "Modal Price"


def load_price_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Prepared price data not found: {DATA_PATH}\n"
            "Run src\\prepare_data.py first."
        )

    data = pd.read_csv(DATA_PATH)
    data.columns = data.columns.astype(str).str.strip()

    required_columns = {
        "State",
        "District",
        "Commodity",
        "Arrival_Date",
        TARGET,
    }
    missing = required_columns - set(data.columns)

    if missing:
        raise ValueError(
            f"The price CSV is missing these columns: {sorted(missing)}"
        )

    data["State"] = data["State"].astype("string").str.strip()
    data = data[
        data["State"].str.casefold().eq("andhra pradesh")
    ].copy()

    data["District"] = data["District"].astype("string").str.strip()
    data["Commodity"] = data["Commodity"].astype("string").str.strip()
    data["Arrival_Date"] = pd.to_datetime(
        data["Arrival_Date"],
        errors="coerce",
    )
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")

    data = data.dropna(
        subset=[
            "District",
            "Commodity",
            "Arrival_Date",
            TARGET,
        ]
    )
    data = data[data[TARGET] > 0].copy()

    data["Month"] = data["Arrival_Date"].dt.month
    data["DayOfYear"] = data["Arrival_Date"].dt.dayofyear

    return data


def build_model() -> Pipeline:
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "categories",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            (
                "numbers",
                "passthrough",
                NUMERIC_FEATURES,
            ),
        ]
    )

    regressor = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=4,
        tree_method="hist",
    )

    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("regressor", regressor),
        ]
    )


def main() -> None:
    data = load_price_data()

    if len(data) < 20:
        raise ValueError(
            f"Only {len(data)} usable price rows were found. "
            "More price records are needed to train and evaluate the model."
        )

    available_dates = sorted(
        data["Arrival_Date"].dt.normalize().unique()
    )

    if len(available_dates) < 2:
        raise ValueError(
            "The price file needs records from at least two different dates "
            "for a time-based training and evaluation split."
        )

    split_position = int(len(available_dates) * 0.8)
    split_position = max(
        1,
        min(split_position, len(available_dates) - 1),
    )
    test_start_date = pd.Timestamp(available_dates[split_position])

    train_data = data[data["Arrival_Date"] < test_start_date].copy()
    test_data = data[data["Arrival_Date"] >= test_start_date].copy()

    if train_data.empty or test_data.empty:
        raise ValueError(
            "Could not create separate earlier and later date groups "
            "for training and evaluation."
        )

    X_train = train_data[FEATURES]
    y_train = train_data[TARGET]
    X_test = test_data[FEATURES]
    y_test = test_data[TARGET]

    # Evaluate on later dates that were not used for fitting.
    evaluation_model = build_model()
    evaluation_model.fit(X_train, y_train)
    predictions = evaluation_model.predict(X_test)

    mae = float(mean_absolute_error(y_test, predictions))
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))

    # Train the model saved for the app using all available price rows.
    final_model = build_model()
    final_model.fit(data[FEATURES], data[TARGET])

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    model_bundle = {
        "model": final_model,
        "features": FEATURES,
        "trained_through": str(data["Arrival_Date"].max().date()),
        "districts_in_training_data": sorted(
            data["District"].dropna().unique().tolist()
        ),
        "commodities_in_training_data": sorted(
            data["Commodity"].dropna().unique().tolist()
        ),
        "price_unit": "rupees per quintal",
        "validation": {
            "method": "latest dates held out",
            "test_start_date": str(test_start_date.date()),
            "rows": int(len(test_data)),
            "mae": mae,
            "rmse": rmse,
        },
    }

    joblib.dump(model_bundle, MODEL_PATH)

    report = (
        "Andhra Pradesh crop price model report\n"
        "=======================================\n"
        f"Usable Andhra Pradesh rows: {len(data):,}\n"
        f"District names with direct price records: "
        f"{data['District'].nunique()}\n"
        f"Crop names with price records: {data['Commodity'].nunique()}\n"
        f"Data starts: {data['Arrival_Date'].min().date()}\n"
        f"Data ends: {data['Arrival_Date'].max().date()}\n"
        f"Validation starts: {test_start_date.date()}\n"
        f"Validation rows: {len(test_data):,}\n"
        f"MAE: ₹{mae:,.2f} per quintal\n"
        f"RMSE: ₹{rmse:,.2f} per quintal\n"
        "\n"
        "Predictions for districts without direct price records are "
        "estimates and may be less reliable.\n"
    )

    REPORT_PATH.write_text(report, encoding="utf-8")

    print(f"Training rows: {len(train_data):,}")
    print(f"Validation rows: {len(test_data):,}")
    print(f"Validation starts: {test_start_date.date()}")
    print(f"MAE: ₹{mae:,.2f} per quintal")
    print(f"RMSE: ₹{rmse:,.2f} per quintal")
    print(f"Districts with direct records: {data['District'].nunique()}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
        