# CIRVIE Incident Classifier

Automatically classifies IT incident tickets into three business labels using a TF-IDF + LinearSVC pipeline — no GPU required.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?logo=scikitlearn)
![Flask](https://img.shields.io/badge/Flask-3.0-black?logo=flask)
![License](https://img.shields.io/badge/license-MIT-green)
![Build](https://github.com/louiscluzel/Inc-classifier/actions/workflows/build_exe.yml/badge.svg)

## What it does

Given an incident description and a few metadata fields (requester, cause, handled-by, urgency), the model predicts:

| Label | Type | Best score |
|---|---|---|
| `INTERVENTION OMV` | Binary (OUI / NON) | 98.5% accuracy |
| `SERVICE` | Target team | 79.0% accuracy |
| `ORIGINE` | Root-cause category | 72.8% accuracy · 75.1% F1 |

A **Windows Excel integration** is also included: a compiled Flask server (`predict_server.exe`) receives requests from a VBA macro, so business users can classify tickets directly from their spreadsheet without installing Python.

## Project structure

```
Inc-classifier/
├── .github/workflows/      CI — builds predict_server.exe on push
├── data/                   Training data (not included — proprietary)
├── models/                 Trained .pkl models (nom_service.json not included)
├── outputs/                Experiment tracking (performance_runs.csv)
├── scripts/                Build and setup scripts (bat, sh, ps1)
├── vba/                    VBA module source (modPredict.bas)
├── main.py                 Train and inference entry point
├── predict_server.py       Flask HTTP server (packaged as .exe for Windows)
├── metrics_logger.py       Logs metrics to outputs/performance_runs.csv
├── get_data.py             Consolidates raw Excel/CSV files into data3.csv
├── requirements.txt        Training dependencies
└── requirements_server.txt Inference-only dependencies (Flask, no torch)
```

## Key features

- **Three independent models** in a single pipeline: OMV (binary), SERVICE (multi-class), ORIGINE (multi-class)
- **Hybrid SERVICE prediction**: ML decides SPECIFIC vs. ALL, then a deterministic JSON lookup maps the requester to their team
- **No GPU needed**: pure scikit-learn sparse pipeline (TF-IDF + OneHotEncoder + LinearSVC)
- **Excel-native delivery**: a PyInstaller `.exe` exposes the model as a local HTTP API consumed by an Excel VBA macro
- **Experiment tracking**: every training run logs metrics to `outputs/performance_runs.csv`

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Train

```bash
python3 main.py
```

Available options:

| Option | Description | Default |
|---|---|---|
| `-m` / `--comment` | Tag this run with a comment | `""` |
| `--test-size` | Test split ratio | `0.2` |
| `--max-features` | TF-IDF vocabulary size | `8000` / `20000` |
| `--c-omv` | LinearSVC regularization C (OMV) | `0.5` |
| `--c-origine` | LinearSVC regularization C (ORIGINE) | `5.0` |

Trained models are saved to `models/`.

> **Data not included**: the `data/` folder and `models/nom_service.json` are gitignored (proprietary). To retrain, provide your own CSV at `data/data3.csv` with columns: `Description`, `INTERVENTION OMV`, `SERVICE`, `ORIGINE`, `Traité par`, `Cause réelle`, `Urgence`, `Demandeur`. Use `get_data.py` to consolidate raw files from `data/raw/`. Supply your own `models/nom_service.json` (format: `{"TEAM_CODE": ["Agent Name", ...]}`)

### Predict (Python API)

```python
from main import predict

result = predict(
    description="Rachat partiel du contrat. Le courrier n'a pas été généré.",
    demandeur="TASSINE-BELLOUT, Anne",
    cause="BASE DE DONNÉES",
    traite="MOA OMVIE",
    urgence="2",
    top_n=3,
)
# {'omv': {'prediction': 'OUI', 'confidence': 98.4, ...},
#  'service': {'prediction': 'SER IND PARIS', ...},
#  'origine': {'prediction': 'BASE DE DONNÉES', ...}}
```

### Excel integration (Windows)

1. Download `predict_server.exe` from the [GitHub Actions artifact](../../actions) (built automatically on every push)
2. Place it in the repo root alongside the `models/` folder
3. Provide your own `data/CIRVIE_INCIDENTS_2026.xlsx` workbook
4. Run `scripts\setup.bat` once to inject the VBA macro
5. Open `data\CIRVIE_INCIDENTS_2026.xlsm` and click **Classify**

### Run the server manually

```bash
pip install -r requirements_server.txt
python3 predict_server.py
# POST http://localhost:8765/predict  {"description": "...", "demandeur": "...", ...}
```

## Pipeline architecture

Each model combines:
- **TF-IDF** on free-text `Description` (bi/tri-grams, sublinear TF)
- **OneHotEncoder** on structured fields: `Cause réelle`, `Traité par`, `Demandeur`, `Urgence`, extracted city
- **LinearSVC** with Platt calibration (for probability estimates)

SERVICE uses a two-stage approach: ML predicts SPECIFIC vs. ALL, then a JSON lookup (`models/nom_service.json`, not included) maps the requester name to their exact team code.

## License

MIT
