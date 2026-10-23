import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from inference_utils import FEATURE_COLUMNS, MODEL_PATH, PASSENGER_ID_COLUMN, PROJECT_DIR


DATA_PATH = PROJECT_DIR / "data" / "train.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
HELDOUT_PATH = OUTPUT_DIR / "heldout_unseen_records.csv"
TARGET_COLUMN = "Survived"
RANDOM_STATE = 42
TEST_SIZE = 0.2
NUMERIC_FEATURES = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
CATEGORICAL_FEATURES = ["Sex", "Embarked"]


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Official Kaggle training data not found at {DATA_PATH}."
        )

    data = pd.read_csv(DATA_PATH)
    required_columns = [TARGET_COLUMN, *FEATURE_COLUMNS]
    if PASSENGER_ID_COLUMN in data.columns:
        required_columns.append(PASSENGER_ID_COLUMN)
    missing_columns = [column for column in required_columns if column not in data]
    if missing_columns:
        raise ValueError(
            "Training CSV is missing required columns: " + ", ".join(missing_columns)
        )
    if data[TARGET_COLUMN].isna().any():
        raise ValueError("Survived contains missing labels.")
    if not set(data[TARGET_COLUMN].unique()).issubset({0, 1}):
        raise ValueError("Survived must contain only binary labels 0 and 1.")

    features = data[FEATURE_COLUMNS].copy()
    target = data[TARGET_COLUMN].astype(int)
    train_features, test_features, train_target, test_target = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model.fit(train_features, train_target)
    predictions = model.predict(test_features)
    test_accuracy = float(accuracy_score(test_target, predictions))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    heldout_records = test_features.copy()
    if PASSENGER_ID_COLUMN in data.columns:
        heldout_records.insert(
            0, PASSENGER_ID_COLUMN, data.loc[test_features.index, PASSENGER_ID_COLUMN]
        )
    heldout_records.to_csv(HELDOUT_PATH, index=False)

    metrics = {
        "dataset": "Official Kaggle Titanic competition train.csv",
        "dataset_rows": int(len(data)),
        "training_rows": int(len(train_features)),
        "testing_rows": int(len(test_features)),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "stratified": True,
        "test_accuracy": test_accuracy,
        "model": "RandomForestClassifier",
        "model_path": MODEL_PATH.name,
        "features": FEATURE_COLUMNS,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print(f"Training rows: {len(train_features)}")
    print(f"Testing rows: {len(test_features)}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved held-out records: {HELDOUT_PATH}")


if __name__ == "__main__":
    main()
