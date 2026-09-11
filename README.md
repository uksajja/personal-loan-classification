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
  - conda env create -f environment.yaml
  - conda activate env-personal-loan
  - open notebook and select the local python .venv as kernel
  - run all the cells

- Inside docker
  - docker build -t personal-loan-classification .
  - docker run --rm -p 8888:8888 -e JUPYTER_TOKEN=my-secret-token personal-loan-classification
  - Run with gpu
    - check if GPU is available
    - nvidia-smi
    - If avialable,
    - docker run --gpus all --rm -p 8888:8888 -e JUPYTER_TOKEN=my-secret-token personal-loan-classification
    - open http://localhost:8888/?token=my-secret-token
    - run the notebook

