import time

from inference_utils import (
    FEATURE_COLUMNS,
    PASSENGER_ID_COLUMN,
    STREAM_RECORD_COUNT,
    STREAM_START_INDEX,
    load_heldout_records,
    load_model,
    prediction_label,
    require_records,
)


def main() -> None:
    model = load_model()
    records = load_heldout_records()
    end_index = STREAM_START_INDEX + STREAM_RECORD_COUNT
    require_records(records, end_index, "Streaming inference")

    stream = records.iloc[STREAM_START_INDEX:end_index]
    for event_number, (_, passenger) in enumerate(stream.iterrows(), start=1):
        passenger_features = passenger.loc[FEATURE_COLUMNS].to_frame().T
        prediction = int(model.predict(passenger_features)[0])
        passenger_id = passenger[PASSENGER_ID_COLUMN]
        print(
            f"Event {event_number}/{STREAM_RECORD_COUNT} | "
            f"PassengerId: {passenger_id} | "
            f"Predicted class: {prediction} ({prediction_label(prediction)})",
            flush=True,
        )
        if event_number < STREAM_RECORD_COUNT:
            time.sleep(1)


if __name__ == "__main__":
    main()
