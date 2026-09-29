## Project Evolution

### v1.0.0 — ML Development

Initial machine learning implementation.

- Exploratory Data Analysis
- Data preprocessing
- Feature engineering
- Model development
- Model comparison
- Hyperparameter tuning
- Evaluation
- Results analysis

## Steps to Run the project:
----------------------------

- Local venv environment
  - py -3.12 -m venv .venv
  - source .venv/Scripts/activate  (Git bash)  or .\.venv\Scripts\Activate.ps1 (PowerShell)
  - source .venv/bin/activate (on linux)
  - python -V
  - python -m pip install --upgrade pip
  - python -m pip install -r requirements.txt
  - open notebook and select the local python .venv as kernel
  - run all the cells

  - deactivate if you want to come out of the virtual environment

- Local conda environemnt
  - conda env create --prefix C:\github\personal-loan-classification\.conda -f .\environment.yaml
  - conda activate env-personal-loan
  - open notebook and select the local python .venv as kernel
  - run all the cells

- Inside docker
  - docker compose --env-file .env config    # validates + shows fully resolved compose file — do this first

### Build all images, including training

- docker compose --profile train build

### Start MLflow

docker compose up -d mlflow

### Run training

docker compose --profile train up training

### Start serving

docker compose up -d serving


- MLflow UI: http://localhost:5000
- Serving API: http://localhost:8080
- Serving health: http://localhost:8080/ping
