import os
import sys
import logging
from pathlib import Path



logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

log_dir = PROJECT_ROOT / "logs"
log_filepath = log_dir / "personal-loan-classifier.log"
log_dir.mkdir(parents=True, exist_ok=True)


logging.basicConfig(
    level= logging.INFO,
    format= logging_str,

    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_filepath)
    ]
)

Logger = logging.getLogger("PersonalLoanClassifierLogger")