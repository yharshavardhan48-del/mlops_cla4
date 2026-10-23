# Titanic ML Deployment Strategies and Model Serving

This project trains one scikit-learn pipeline on the labelled training portion of the official Kaggle Titanic competition `train.csv`, then reuses the exact saved `model.pkl` for online, batch, streaming, and FastAPI inference. Kaggle's `test.csv` is not used.

Official dataset: [Kaggle Titanic competition data](https://www.kaggle.com/competitions/titanic/data). Download only `train.csv` and place it in `data\train.csv`.

## Project structure

```text
data\
  train.csv
outputs\
  heldout_unseen_records.csv
  metrics.json
screenshots\
train_model.py
inference_utils.py
online_inference.py
batch_inference.py
streaming_inference.py
app.py
client.py
generate_report.py
requirements.txt
README.md
report.pdf
model.pkl
batch_predictions.csv
```

The submission ZIP includes the official CSV, saved model, inference outputs, and report. Python caches, virtual environments, and local credentials are excluded. The raw Kaggle CSV is excluded from GitHub publishing by `.gitignore`; download the official `train.csv` into `data\` locally before running training. Confirm Kaggle's competition terms before redistributing its data.

## Windows installation

From PowerShell in this project folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The project was validated with Python 3.11.9. The initially selected Python 3.14 environment crashed on exit when importing the installed scikit-learn/joblib native modules, so the project is tested in an isolated Python 3.11 environment. The scikit-learn version is pinned to match the serialized pipeline.

Before training, confirm the official `data\train.csv` is present. Train the complete pipeline once:

```powershell
python train_model.py
```

This creates `model.pkl`, `outputs\metrics.json`, and `outputs\heldout_unseen_records.csv`. The training script uses a stratified 80:20 split with random seed 42. All imputers and one-hot encoding are fitted only on the training split. The measured held-out accuracy is **82.12%** (147 correct of 179 held-out passengers; exact accuracy: 0.821229).

## Inference

Each script loads the existing `model.pkl`; none retrains the model:

```powershell
python online_inference.py
python batch_inference.py
python streaming_inference.py
```

Online inference uses the first held-out passenger; batch uses the next 20 records and writes `batch_predictions.csv`; streaming processes the subsequent 10 records one at a time, with a one-second pause between events. These three subsets do not overlap.

## FastAPI model serving

Start the API in one terminal:

```powershell
python -m uvicorn app:app --reload
```

Open the interactive Swagger UI at <http://127.0.0.1:8000/docs>. Health is available at <http://127.0.0.1:8000/health>.

Example valid request:

```json
{
  "Pclass": 3,
  "Sex": "male",
  "Age": 22,
  "SibSp": 1,
  "Parch": 0,
  "Fare": 7.25,
  "Embarked": "S"
}
```

Age, Fare, and Embarked may be omitted or set to `null`; the saved pipeline imputes missing values. Invalid passenger classes, out-of-range numbers, unsupported categorical values, and unknown fields receive HTTP 422 validation responses.

Example invalid request (Pclass 4; expected HTTP 422):

```json
{
  "Pclass": 4,
  "Sex": "male",
  "Age": 22,
  "SibSp": 1,
  "Parch": 0,
  "Fare": 7.25,
  "Embarked": "S"
}
```

With the server running, use the requests-based client in another terminal:

```powershell
python client.py
```

The response includes the predicted class, a readable label, and class probabilities when provided by the classifier. The API loads the saved model once during startup. Validated results: `/health` returned **200**, a valid `/predict` returned **200**, an invalid `Pclass: 4` returned **422**, `/docs` returned **200**, and `client.py` received **200**. Optional Age/Fare/Embarked values also passed inference; out-of-range numeric and unsupported categorical values returned **422**.

## Generated files and validation

- `model.pkl`: fitted preprocessing and classification pipeline.
- `outputs\metrics.json`: split details and measured held-out accuracy.
- `outputs\heldout_unseen_records.csv`: held-out feature rows without labels.
- `batch_predictions.csv`: exactly 20 batch predictions.
- `outputs\validation_results.json`: observed endpoint and inference validation results.
- `report.pdf`: concise report populated with measured results and endpoint test statuses.
- `screenshots\health.png`, `screenshots\predict_success.png`, and `screenshots\predict_422.png`: genuine screenshots captured from the running Swagger/API pages.
- `screenshots\online_inference.png`, `screenshots\batch_inference.png`, `screenshots\streaming_inference.png`, and `screenshots\client.png`: terminal screenshots to capture manually; desktop-terminal screenshot capture was not available through the browser capture tool.

Run scripts and API tests in the order listed in the assignment. Recreate the PDF from the measured metrics and test results with:

```powershell
python generate_report.py
```

Capture genuine terminal output for online, batch, streaming inference, and the Python client. Do not put credentials in the repository.

## Publish to GitHub

1. Create an empty GitHub repository and copy its HTTPS or SSH URL.
2. From PowerShell in this folder, initialize and publish the project:

   ```powershell
   git init
   git add .
   git commit -m "Complete Titanic model deployment assignment"
   git branch -M main
   git remote add origin <your-repository-url>
   git push -u origin main
   ```

3. Review the staged file list before committing. Never publish Kaggle API tokens, `.env` files, passwords, or private credentials. The repository intentionally excludes `data\train.csv`; the submission ZIP contains it as required by the assignment.
