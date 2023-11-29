"""Script to create the seed file for column-level descriptions

It extracts the table and column names, adds some boilerplate template for descriptions where possible,
and then outputs a csv file.
The file can then be manually filled in.
"""
import sys

import pandas as pd

from config import DB_FILE, TABLE_DESCRIPTION_FILE
from sql_common import SQLExecutor


def main():
    if len(sys.argv) != 2:
        raise ValueError("Usage: <script> <ouput-csv>")
    output_path = sys.argv[1]

    executor = SQLExecutor(DB_FILE, TABLE_DESCRIPTION_FILE)
    tables, _ = executor.execute(
        "SELECT name FROM sqlite_schema WHERE type ='table' AND name NOT LIKE 'sqlite_%';"
    )
    assert tables is not None
    table_columns = []

    for table in tables:
        table = table[0]
        fields, _ = executor.execute(f"PRAGMA table_info(`{table}`)")
        for field in fields:
            table_columns.append((table, field[1], field[2]))
    df = pd.DataFrame(table_columns, columns=["Table", "Col_Name", "Col_Type"])

    def desc_map(table_name: str, col_name: str):
        if "_id" in col_name:
            ref_table = col_name.split("_id")[0]
            ref_table = "Responses" if ref_table == "Response" else ref_table
            return f"Foreign Key ({col_name}) references {ref_table}(id)"
        if col_name == "id":
            return f"Unique ID for {table_name} instances"
        if col_name == "Name":
            return executor.table_desc[table_name]
        return ""

    df["Description"] = df.apply(lambda x: desc_map(x.Table, x.Col_Name), axis=1)
    print(df)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
