"""Main script to create annotations (i.e. the SQL queries)

Provides functionality to switch between the different approaches.
Also runs the SQL queries to include the output result.
"""
import argparse
from pathlib import Path

from multiprocessing_on_dill import Pool
from tqdm import tqdm

from config import COLUMN_DESCRIPTION_FILE, DB_FILE, RAW_DESCRIPTION_FILE
from eval import load_eval_csv, load_eval_json, save_eval
from retrieval import PerfectRetriever, load_retriever_generator
from sql_common import SQLExecutor
from sql_generation import load_base_generator
from sql_generation_agents import load_agent


def main():
    parser = argparse.ArgumentParser(description="Generate sql for an eval set")
    parser.add_argument("-i", "--input", required=True, help="Input file")
    parser.add_argument("-o", "--output", required=True, help="Output file")
    parser.add_argument(
        "-g",
        "--generator",
        choices=("base", "agent", "retriever"),
        default="base",
        help="The SQLGenerator",
    )
    parser.add_argument(
        "-d",
        "--description",
        choices=("none", "per-column"),
        default="none",
        help="The description added to schema",
    )
    parser.add_argument(
        "-r",
        "--raw-description",
        action="store_true",
        help="Include original raw description",
    )
    parser.add_argument(
        "-s",
        "--few-shot",
        action="store_true",
        help="Include few shot examples. Only applies for base generator.",
    )
    parser.add_argument(
        "-f",
        "--force-run",
        action="store_true",
        help="Force run all predictions even if they already exist",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    descriptions_file = None
    if args.description == "per-column":
        descriptions_file = COLUMN_DESCRIPTION_FILE

    sql_executor = SQLExecutor(file=DB_FILE, descriptions_file=descriptions_file)

    raw_desc_path = RAW_DESCRIPTION_FILE if args.raw_description else None

    if input_path.suffix == ".csv":
        eval_set = load_eval_csv(input_path, sql_executor)
    else:
        eval_set = load_eval_json(input_path)

    if args.generator == "base":
        if args.description == "none" or raw_desc_path is not None:
            schema = "\n".join(sql_executor.get_table_reprs(num_examples=3).values())
        else:
            # When the description is included we also opt for including distinct values
            schema = "\n".join(
                sql_executor.get_table_reprs(num_examples=0, num_distinct=20).values()
            )

        generator = load_base_generator(
            schema=schema, raw_desc_path=raw_desc_path, use_few_shot=args.few_shot
        )
    elif args.generator == "agent":
        generator = load_agent(DB_FILE)
    elif args.generator == "retriever":
        # Use a perfect retriever
        retriever = PerfectRetriever(
            schema_docs=sql_executor.get_table_reprs(num_examples=0, num_distinct=20),
            eval_set=eval_set,
        )
        generator = load_retriever_generator(sql_executor, retriever=retriever)

    for example in tqdm(eval_set):
        if example.prediction is None and not args.force_run:
            example.generate_prediction(generator)
        # Defensively save the output
        save_eval(eval_set, output_path)

    # Save with queries
    save_eval(eval_set, output_path)

    with Pool(4) as p:
        eval_set = list(
            tqdm(
                p.imap(lambda sample: sample.execute_query(sql_executor), eval_set),
                total=len(eval_set),
            )
        )

    # Save with queries and results
    save_eval(eval_set, output_path)


if __name__ == "__main__":
    main()
