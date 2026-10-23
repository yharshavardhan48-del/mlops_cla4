import pandas as pd

from inference_utils import (
    BATCH_RECORD_COUNT,
    BATCH_START_INDEX,
    FEATURE_COLUMNS,
    HELDOUT_PATH,
    PASSENGER_ID_COLUMN,
    load_heldout_records,
    load_model,
    prediction_label,
    require_records,
)


OUTPUT_PATH = HELDOUT_PATH.parent.parent / "batch_predictions.csv"


def main() -> None:
    model = load_model()
    records = load_heldout_records()
    end_index = BATCH_START_INDEX + BATCH_RECORD_COUNT
    require_records(records, end_index, "Batch inference")

    batch = records.iloc[BATCH_START_INDEX:end_index].copy()
    predictions = model.predict(batch[FEATURE_COLUMNS])

    output = batch.copy()
    output["PredictedSurvived"] = predictions.astype(int)
    output["PredictionLabel"] = [
        prediction_label(prediction) for prediction in predictions
    ]
    output.to_csv(OUTPUT_PATH, index=False)

    saved_records = pd.read_csv(OUTPUT_PATH)
    if len(saved_records) != BATCH_RECORD_COUNT:
        raise RuntimeError(
            f"Expected {BATCH_RECORD_COUNT} predictions in {OUTPUT_PATH}, "
            f"found {len(saved_records)}."
        )

    print(saved_records.to_string(index=False))
    print(f"Verified predictions saved: {len(saved_records)}")
    print(f"Output CSV: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
