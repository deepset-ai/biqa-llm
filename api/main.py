from typing import Optional
import logging
import os

from fastapi import FastAPI
from pydantic import BaseModel

from api.utils import format_markdown_table

from config import DB_FILE, COLUMN_DESCRIPTION_FILE
from sql_common import SQLExecutor
from sql_generation import load_base_generator


LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
logging.basicConfig(format="[%(levelname)s] %(asctime)s %(message)s", level=LOG_LEVEL)
logger = logging.getLogger(__name__)


app = FastAPI()

executor = SQLExecutor(file=DB_FILE, descriptions_file=COLUMN_DESCRIPTION_FILE)
schema = "\n".join(executor.get_table_reprs(num_distinct=20).values())
generator = load_base_generator(
    schema,
    use_few_shot=True
)


class QueryRequest(BaseModel):
    query: str
    params: Optional[dict] = None
    debug: Optional[bool] = False


@app.get("/initialized")
async def check_status():
    return True


@app.post("/query")
async def query(request: QueryRequest):
    logger.info(f"Query: {str(request.query)}")
    sql = generator.generate(str(request.query))
    results, headers = executor.execute(sql)

    answer = format_markdown_table(results, headers)
    answer += f"\n\n**Generated SQL:**\n```sql\n{sql}\n```"
    logger.info(f"SQL: \n{sql}")
    
    return {
        'answer': answer,
        'meta': {'generated_sql': sql},
    }
