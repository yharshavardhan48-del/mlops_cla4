from inference_utils import (
    FEATURE_COLUMNS,
    ONLINE_RECORD_INDEX,
    load_heldout_records,
    load_model,
    prediction_label,
    require_records,
)


def main() -> None:
    model = load_model()
    records = load_heldout_records()
    require_records(records, ONLINE_RECORD_INDEX + 1, "Online inference")

    passenger = records.iloc[[ONLINE_RECORD_INDEX]]
    prediction = int(model.predict(passenger[FEATURE_COLUMNS])[0])

    print("Passenger details:")
    print(passenger.to_string(index=False))
    print(f"Predicted class: {prediction}")
    print(f"Prediction: {prediction_label(prediction)}")


if __name__ == "__main__":
    main()
