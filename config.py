import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if OPENAI_API_KEY == "":
    logger.warning("OPENAI_API_KEY not set")


DATA_DIR = Path("data/")
DB_FILE = DATA_DIR / "survey_results_normalized_v2.db"
TABLE_DESCRIPTION_FILE = DATA_DIR / "survey_tables_descriptions_v2.csv"
COLUMN_DESCRIPTION_FILE = DATA_DIR / "survey_columns_descriptions_v2.csv"
RAW_DESCRIPTION_FILE = DATA_DIR / "survey_results_schema_v2.csv"
