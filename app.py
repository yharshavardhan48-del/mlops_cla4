from contextlib import asynccontextmanager
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from inference_utils import FEATURE_COLUMNS, load_model, prediction_label


class PassengerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    Pclass: Literal[1, 2, 3]
    Sex: Literal["female", "male"]
    Age: float | None = Field(default=None, ge=0, le=120)
    SibSp: int = Field(ge=0, le=20)
    Parch: int = Field(ge=0, le=20)
    Fare: float | None = Field(default=None, ge=0)
    Embarked: Literal["C", "Q", "S"] | None = None


@asynccontextmanager
async def lifespan(application: FastAPI):
    application.state.model = load_model()
    yield
    del application.state.model


app = FastAPI(
    title="Titanic Survival Prediction API",
    description="Serves predictions from the saved Titanic preprocessing and model pipeline.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request) -> dict[str, object]:
    model_loaded = getattr(request.app.state, "model", None) is not None
    return {
        "status": "ok" if model_loaded else "unavailable",
        "model_loaded": model_loaded,
    }


@app.post("/predict")
def predict(passenger: PassengerRequest, request: Request) -> dict[str, object]:
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    values = passenger.model_dump()
    for column in ("Age", "Fare", "Embarked"):
        if values[column] is None:
            values[column] = np.nan
    features = pd.DataFrame(
        [[values[column] for column in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS,
    )
    predicted_class = int(model.predict(features)[0])

    response: dict[str, object] = {
        "predicted_class": predicted_class,
        "prediction": prediction_label(predicted_class),
    }
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        response["class_probabilities"] = {
            str(class_label): float(probability)
            for class_label, probability in zip(model.classes_, probabilities)
        }
    return response
