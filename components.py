import sqlite3
from typing import Union

from haystack.preview import component


@component
class SQLExecutorComponent:
    """
    Wraps the SQLExecutor as a haystack component
    """

    def __init__(self, executor) -> None:
        self.executor = executor

    def get_table_reprs(self, *args, **kwargs):
        return self.executor.get_table_reprs(*args, **kwargs)

    @component.output_types(result=Union[list, str])
    def run(self, query: str):
        try:
            result, _ = self.executor.execute(query)
            return {"result": result}
        except sqlite3.OperationalError as e:
            return {"result": e.args[0]}
