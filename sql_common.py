import sqlite3
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

EMPTY_QUERY = "SELECT NULL LIMIT 0;"


class SQLExecutor:
    """
    Takes in an SQL query and executes it on an SQLite DB
    """

    def __init__(self, file: Path, descriptions_file: Path = None) -> None:
        self.file = file
        self.col_desc = {}
        if descriptions_file is not None:
            df = pd.read_csv(descriptions_file)
            df["Table-Column"] = df.apply(lambda x: f"{x.Table}-{x.Col_Name}", axis=1)
            self.col_desc: dict = df.set_index("Table-Column")["Description"].to_dict()

    def execute(self, query: str) -> Optional[list[tuple]]:
        con = sqlite3.connect(f"file:{self.file.resolve()}?mode=ro", uri=True)
        try:
            res = con.execute(query)
            headers = list(map(lambda x: x[0], res.description))
            return res.fetchall(), headers
        except sqlite3.OperationalError as ex:
            print(ex)
            return None, None

    def get_table_reprs(
        self, num_examples: int = 0, num_distinct: int = 0, only_desc: bool = False
    ) -> Dict[str, str]:
        tables, _ = self.execute(
            "SELECT name FROM sqlite_schema WHERE type ='table' AND name NOT LIKE 'sqlite_%';"
        )
        assert tables is not None
        schema = {}
        for table in tables:
            table = table[0]
            fields, _ = self.execute(f"PRAGMA table_info(`{table}`)")
            assert fields is not None
            fields_str = "\n"
            for f in fields:
                field_name, field_type = f[1], f[2]
                field_is_id = (field_name == "id") or ("_id" in field_name)

                field_desc = self.col_desc.get(f"{table}-{field_name}", "")

                if only_desc:
                    # When in only_desc mode, skip id fields (not informative)
                    if field_is_id:
                        continue
                    fields_str += f"{field_desc}\n"
                else:
                    field_desc = f" -- {field_desc}" if field_desc != "" else ""
                    fields_str += f"{field_name}: {field_type}{field_desc}\n"

                # Only include distinct info for non-id fields
                if num_distinct != 0 and not field_is_id:
                    distinct_items, _ = self.execute(
                        f"SELECT DISTINCT {field_name} from {table};"
                    )
                    distinct_items = [d[0] for d in distinct_items]
                    # If there are very many distinct values, don't include any distinct values
                    # Possibly the field is just unique-per-row and so not worth it
                    if len(distinct_items) > 100:
                        continue
                    fields_str += f"Distinct: {distinct_items[:num_distinct]}\n"

            examples = ""
            if num_examples != 0:
                cells, _ = self.execute(f"SELECT * FROM `{table}` LIMIT {num_examples}")
                assert cells is not None
                cells = [[str(cell) for cell in row] for row in cells]
                cells = ["\t".join(row) for row in cells]
                examples = "\n" + "\n".join(cells)
            schema[table] = f"`{table}`{fields_str}{examples}"
        return schema
