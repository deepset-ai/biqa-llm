from typing import Any, Dict, List, Optional, Union

from canals.errors import PipelineRuntimeError
from canals.serialization import default_from_dict, default_to_dict
from haystack.preview import Pipeline, component
from loguru import logger
from sentence_transformers import SentenceTransformer

from components import SQLExecutorComponent
from config import COLUMN_DESCRIPTION_FILE, DB_FILE
from eval import SQLSample
from sql_common import EMPTY_QUERY, SQLExecutor
from sql_generation import OpenAI


def find_tables_from_query(tables, query):
    query_tables = []

    for t in tables:
        if t.lower() in query.lower():
            query_tables.append(t)

    return set(query_tables)


@component
class PerfectRetriever:
    def __init__(self, schema_docs: Dict[str, str], eval_set: List[SQLSample]) -> None:
        self.schema_docs = schema_docs
        self.eval_set = eval_set
        self.question_to_tables = {
            e.question: list(
                find_tables_from_query(self.schema_docs.keys(), e.labels[0].query)
            )
            for e in eval_set
        }

    def to_dict(self) -> Dict[str, Any]:
        return default_to_dict(
            self, schema_docs=self.schema_docs, eval_set=self.eval_set
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PerfectRetriever":
        return default_from_dict(cls, data)

    @component.output_types(docs=list[str])
    def run(self, query: str):
        question_tables = self.question_to_tables[query]
        table_docs = [self.schema_docs[t] for t in question_tables]
        return {"docs": table_docs}


@component
class Retriever:
    def __init__(
        self,
        docs: List[str],
        model: str,
        embedded_docs: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> None:
        embedded_docs = embedded_docs or docs
        self.model = SentenceTransformer(model)
        self.model_name = model
        self.docs = docs
        self.embeddings = self.model.encode(embedded_docs, normalize_embeddings=True)
        self.top_k = top_k

    def to_dict(self) -> Dict[str, Any]:
        return default_to_dict(
            self, docs=self.docs, model=self.model_name, top_k=self.top_k
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Retriever":
        return default_from_dict(cls, data)

    @component.output_types(docs=list[str])
    def run(self, query: str, top_k: Optional[int] = None):
        top_k = top_k or self.top_k
        embedding = self.model.encode(query, normalize_embeddings=True)
        similiarities = self.embeddings @ embedding[:, None]
        similiarities = similiarities[:, 0]
        indices = similiarities.argpartition(-top_k)[-top_k:]
        docs = [self.docs[index] for index in indices]
        return {"docs": docs}


@component
class PromptBuilder:
    def to_dict(self) -> Dict[str, Any]:
        return default_to_dict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptBuilder":
        return default_from_dict(cls, data)

    @component.output_types(prompt=str)
    def run(self, query: str, docs: List[str]):
        docs_ = "\n".join(docs)
        prompt = f"""Please return an SQLite query that answers the given question.
The following is the schema of the database containing a survey result tables with some example rows or field-wise distinct values:
{docs_}
Please return an SQLite query that answers the following question. Account for NULL values. If you need to make assumptions, do not state them and do not explain your query in any other way. Please make sure to disregard null entries: {query}"""  # noqa
        return {"prompt": prompt}


@component
class PromptNode:
    """
    Simple prompt node implementation.
    There seem to be many considerations for the eventual version for v2, but this should be good enough for now.
    """

    def __init__(
        self,
        model,
        prompt: str,
        temperature: float = 1.0,
        max_tokens: int = 300,
        stop: Optional[str] = None,
    ) -> None:
        self.prompt = prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stop = stop
        self.model = model

    def to_dict(self) -> Dict[str, Any]:
        return default_to_dict(
            self,
            prompt=self.prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=self.stop,
            model=self.model.to_dict(),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptNode":
        return default_from_dict(cls, data)

    @component.output_types(generated=str)
    def run(
        self,
        prepend: Optional[str] = "",
        append: Optional[str] = "",
        prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
    ):
        prompt = prompt or self.prompt
        temperature = temperature or self.temperature
        max_tokens = max_tokens or self.max_tokens
        stop = stop or self.stop
        return {
            "generated": self.model.complete(
                prepend + prompt + append,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop,
            )
        }


def generation_pipeline(
    sql_executor_: SQLExecutor,
    eval_: bool = False,
    retriever: Optional[Union[Retriever, PerfectRetriever]] = None,
) -> Pipeline:
    pipeline = Pipeline()

    sql_executor = SQLExecutorComponent(sql_executor_)

    if retriever is None:
        retriever = Retriever(
            list(sql_executor.get_table_reprs(num_examples=3).values()),
            "sentence-transformers/all-mpnet-base-v2",
            list(sql_executor.get_table_reprs(num_examples=3).values()),
            top_k=40,
        )
    query_generator = PromptNode(OpenAI("gpt-4"), "", temperature=0)
    prompt_builder = PromptBuilder()
    pipeline.add_component("query_generator", query_generator)
    pipeline.add_component("retriever", retriever)
    pipeline.add_component("prompt_builder", prompt_builder)
    pipeline.connect("retriever.docs", "prompt_builder.docs")
    pipeline.connect("prompt_builder.prompt", "query_generator.prepend")

    if not eval_:
        pipeline.add_component("sql_executor", sql_executor)
        pipeline.connect("query_generator.generated", "sql_executor.query")
    return pipeline


class RetrieverSQLGenerator:
    def __init__(self, pipeline) -> None:
        self.pipeline = pipeline

    def generate(self, question: str):
        try:
            result = self.pipeline.run(
                {
                    "prompt_builder": {"query": question},
                    "retriever": {"query": question},
                }
            )
            return result["query_generator"]["generated"]
        except Exception as e:
            traceback.print_exception(e)
            logger.warning("".join(traceback.format_tb(tb=e.__traceback__)))
            return EMPTY_QUERY


def load_retriever_generator(
    sql_executor: SQLExecutor,
    retriever: Optional[Union[Retriever, PerfectRetriever]] = None,
):
    pipeline = generation_pipeline(
        sql_executor_=sql_executor, eval_=True, retriever=retriever
    )
    return RetrieverSQLGenerator(pipeline)


if __name__ == "__main__":
    executor = SQLExecutor(DB_FILE, descriptions_file=COLUMN_DESCRIPTION_FILE)
    pipeline = generation_pipeline(executor)
    query = (
        "What are the percentages of various educational attainments among respondents?"
    )
    try:
        print(
            pipeline.run(
                {"prompt_builder": {"query": query}, "retriever": {"query": query}}
            )
        )
    except PipelineRuntimeError as e:
        print("Pipeline Error. Possibly SQL Syntax error")
        import traceback

        traceback.print_exc()
