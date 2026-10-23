import json
from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from inference_utils import PROJECT_DIR


METRICS_PATH = PROJECT_DIR / "outputs" / "metrics.json"
VALIDATION_PATH = PROJECT_DIR / "outputs" / "validation_results.json"
REPORT_PATH = PROJECT_DIR / "report.pdf"


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required report input is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_report() -> None:
    metrics = load_json(METRICS_PATH)
    validation = load_json(VALIDATION_PATH)
    accuracy = float(metrics["test_accuracy"])
    test_rows = int(metrics["testing_rows"])
    correct = round(accuracy * test_rows)
    api_tests = [
        ("GET /health", validation["health"], 200),
        ("POST /predict (valid request)", validation["valid_predict"], 200),
        ("POST /predict (Pclass = 4)", validation["invalid_predict"], 422),
        ("GET /docs", validation["swagger_docs"], 200),
        ("Python client", validation["client_status_code"], 200),
    ]

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            alignment=TA_CENTER,
            textColor=colors.HexColor("#183153"),
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionHeading",
            parent=styles["Heading2"],
            textColor=colors.HexColor("#183153"),
            spaceBefore=8,
            spaceAfter=4,
        )
    )

    document = SimpleDocTemplate(
        str(REPORT_PATH),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
        title="Titanic ML Deployment Strategies and Model Serving",
        author="",
    )
    story = [
        Paragraph(
            "ML Model Deployment Strategies and Model Serving using FastAPI",
            styles["ReportTitle"],
        ),
        Paragraph(f"College assignment report | {date.today().isoformat()}", styles["Normal"]),
        Spacer(1, 4 * mm),
        Paragraph("Objective and dataset", styles["SectionHeading"]),
        Paragraph(
            "This project trains and serves a passenger survival classifier using only "
            "the official Kaggle Titanic competition labelled train.csv "
            "(891 records). Kaggle test.csv was not used.",
            styles["BodyText"],
        ),
        Paragraph("Preprocessing and model", styles["SectionHeading"]),
        Paragraph(
            "The selected inputs are Pclass, Sex, Age, SibSp, Parch, Fare, and "
            "Embarked. Numeric missing values are imputed with the training-set "
            "median; categorical missing values use the training-set most-frequent "
            "value, followed by one-hot encoding. A ColumnTransformer and Pipeline "
            "keep preprocessing with a 300-tree RandomForestClassifier. The complete "
            "fitted pipeline is saved as model.pkl using Joblib. Preprocessing is "
            "fitted only on the training portion.",
            styles["BodyText"],
        ),
        Paragraph("Training and evaluation", styles["SectionHeading"]),
        Paragraph(
            f"A stratified 80:20 split with random_state=42 produced "
            f"{metrics['training_rows']} training and {test_rows} held-out records. "
            f"Measured accuracy was {accuracy:.4%} ({correct}/{test_rows} correct). "
            "The held-out feature rows were exported without survival labels for "
            "unseen-record inference.",
            styles["BodyText"],
        ),
        Paragraph("Inference strategies", styles["SectionHeading"]),
        Paragraph(
            "Online inference predicts one held-out passenger per call. Batch "
            "inference sends 20 different held-out passengers through one prediction "
            "call and writes batch_predictions.csv. Streaming simulates ten separate "
            "incoming passenger events, predicting each individually with a "
            "one-second pause between events. The three subsets are disjoint.",
            styles["BodyText"],
        ),
        Paragraph("FastAPI serving and model reuse", styles["SectionHeading"]),
        Paragraph(
            "FastAPI provides a lightweight HTTP interface, typed Pydantic request "
            "validation, automatic OpenAPI documentation, and a health endpoint. "
            "The application loads model.pkl once at startup. Every inference mode "
            "uses the same fitted artifact, keeping preprocessing and predictions "
            "consistent and preventing retraining or data leakage during serving.",
            styles["BodyText"],
        ),
        Paragraph("API validation and test results", styles["SectionHeading"]),
    ]

    table_data = [["Check", "Observed status", "Expected"]]
    for name, observed, expected in api_tests:
        table_data.append([name, str(observed), str(expected)])
    table = Table(table_data, colWidths=[83 * mm, 38 * mm, 38 * mm], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#183153")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9AA9B8")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EEF3F8")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.extend(
        [
            table,
            Spacer(1, 3 * mm),
            Paragraph(
                f"Additional checks: optional Age/Fare/Embarked request returned HTTP "
                f"{validation['optional_fields_predict']}; invalid Age, Sex, and "
                f"SibSp values each returned HTTP 422. The client returned HTTP "
                f"{validation['client_status_code']}.",
                styles["BodyText"],
            ),
            Paragraph("Conclusion", styles["SectionHeading"]),
            Paragraph(
                "The assignment demonstrates a leakage-aware train/test workflow and "
                "three deployment patterns backed by one persisted pipeline. The "
                "FastAPI endpoint validates passenger input and returns a class, "
                "readable label, and class probabilities.",
                styles["BodyText"],
            ),
        ]
    )
    document.build(story)
    print(f"Report created from measured outputs: {REPORT_PATH}")


if __name__ == "__main__":
    build_report()
