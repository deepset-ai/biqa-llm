"""Count tokens (or print out the prompt)
"""
import argparse

import tiktoken

from config import COLUMN_DESCRIPTION_FILE, DB_FILE, RAW_DESCRIPTION_FILE
from sql_common import SQLExecutor
from sql_generation import load_base_generator


def main():
    parser = argparse.ArgumentParser(description="Generate sql for an eval set")
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
        "-p", "--print-prompt", action="store_true", help="Print out the prompt"
    )

    args = parser.parse_args()

    descriptions_file = None
    if args.description == "per-column":
        descriptions_file = COLUMN_DESCRIPTION_FILE

    sql_executor = SQLExecutor(file=DB_FILE, descriptions_file=descriptions_file)

    raw_desc_path = RAW_DESCRIPTION_FILE if args.raw_description else None
    if args.description == "none" or raw_desc_path is not None:
        schema = "\n".join(sql_executor.get_table_reprs(num_examples=3).values())
    else:
        # When the description is included we also opt for including distinct values
        schema = "\n".join(
            sql_executor.get_table_reprs(num_examples=0, num_distinct=20).values()
        )

    generator = load_base_generator(
        schema=schema,
        raw_desc_path=raw_desc_path,
        use_few_shot=args.few_shot,
    )

    encoding = tiktoken.encoding_for_model("gpt-4")
    prompt_tokens = encoding.encode(generator.prompt)
    if args.print_prompt:
        print("\n" + generator.prompt + "\n")
    print("Num Tokens: ", len(prompt_tokens))


if __name__ == "__main__":
    main()
