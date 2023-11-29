from typing import List


def format_table(data: List[List]):
    if not data:
        return ""

    # Calculate the width of each column
    column_widths = [max(len(str(item)) for item in column) for column in zip(*data)]

    # Create a separator line
    separator = f"+-{'-+-'.join('-' * width for width in column_widths)}-+"

    # Format each row and combine into a table string
    table_string = separator + "\n"
    for row in data:
        row_internal_str = " | ".join(str(item).ljust(width) for item, width in zip(row, column_widths))
        row_string = f"| {row_internal_str} |\n"
        table_string += row_string + separator + "\n"

    return table_string


def format_markdown_table(data: List[List], headers: List[str]) -> str:
    if not data or len(data) == 0:
        return ""

    row_strings = [
        "|" + "|".join(headers) + "|",
        "|" + "|".join(['---'] * len(data[0])) + "|"
    ]
    for row in data:
        row_strings.append(
            "|" + "|".join(str(item) for item in row) + "|"
        )
    return "\n".join(row_strings)
