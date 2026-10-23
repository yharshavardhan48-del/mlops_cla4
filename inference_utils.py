from pathlib import Path

import joblib
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "model.pkl"
HELDOUT_PATH = PROJECT_DIR / "outputs" / "heldout_unseen_records.csv"
FEATURE_COLUMNS = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
PASSENGER_ID_COLUMN = "PassengerId"
ONLINE_RECORD_INDEX = 0
BATCH_START_INDEX = 1
BATCH_RECORD_COUNT = 20
STREAM_START_INDEX = BATCH_START_INDEX + BATCH_RECORD_COUNT
STREAM_RECORD_COUNT = 10


def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Saved model not found at {MODEL_PATH}. Run train_model.py first."
        )
    return joblib.load(MODEL_PATH)


def load_heldout_records() -> pd.DataFrame:
    if not HELDOUT_PATH.is_file():
        raise FileNotFoundError(
            f"Held-out records not found at {HELDOUT_PATH}. Run train_model.py first."
        )

    records = pd.read_csv(HELDOUT_PATH)
    required_columns = [PASSENGER_ID_COLUMN, *FEATURE_COLUMNS]
    missing_columns = [column for column in required_columns if column not in records]
    if missing_columns:
        raise ValueError(
            "Held-out records are missing required columns: "
            + ", ".join(missing_columns)
        )
    if records[PASSENGER_ID_COLUMN].duplicated().any():
        raise ValueError("Held-out records contain duplicate PassengerId values.")

    return records


def prediction_label(prediction: int) -> str:
    return "Survived" if int(prediction) == 1 else "Did not survive"


def require_records(records: pd.DataFrame, count: int, purpose: str) -> None:
    if len(records) < count:
        raise ValueError(
            f"{purpose} requires {count} held-out records, but only "
            f"{len(records)} are available."
        )
