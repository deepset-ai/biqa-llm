import csv
from collections import defaultdict
from dataclasses import asdict, dataclass
from json import dump, load
from pathlib import Path
from typing import Optional

from sqlglot import exp, parse_one

from sql_generation import SQLExecutor, SQLGenerator


@dataclass
class SQLQuery:
    query: str
    results: Optional[list[tuple]] = None

    def execute_query(self, executor: SQLExecutor):
        if self.results:
            return self
        self.results, _ = executor.execute(self.query)
        return self

    def strip(self):
        self.results = None


@dataclass
class SQLSample:
    question: str
    labels: list[SQLQuery]
    prediction: Optional[SQLQuery] = None
    pred_eval: str = ""
    comment: str = ""

    def generate_prediction(self, generator: SQLGenerator):
        query = generator.generate(self.question)
        self.prediction = SQLQuery(query=query)
        return self

    def execute_query(self, executor: SQLExecutor):
        for label in self.labels:
            label.execute_query(executor)
        if self.prediction:
            self.prediction.execute_query(executor)
        return self

    def check(self, fuzzy: bool = True):
        for label in self.labels:
            if not fuzzy:
                if label.results == self.prediction.results:
                    return True
                continue
            if label.results is None:
                return False
            answer_cells = set(cell for row in label.results for cell in row)
            predicted_cells = set(
                cell for row in self.prediction.results for cell in row
            )
            if answer_cells.issubset(predicted_cells):
                return True
        return False

    def strip(self):
        for label in self.labels:
            label.strip()
        self.prediction = None


def load_eval_csv(path: Path, executor: SQLExecutor) -> list[SQLSample]:
    dataset = []
    with path.open() as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            simple_question = row[2]
            sql_query = row[4]
            results = executor.execute(sql_query)
            dataset.append(
                SQLSample(
                    question=simple_question,
                    labels=[SQLQuery(query=sql_query, results=results)],
                )
            )
    return dataset


def load_eval_json(path: Path) -> list[SQLSample]:
    with path.open() as f:
        eval_set = load(f)
    return [
        SQLSample(
            question=sample["question"],
            labels=[dict_to_sql_query(label) for label in sample["labels"]],
            prediction=(
                None
                if sample["prediction"] is None
                else dict_to_sql_query(sample["prediction"])
            ),
            pred_eval=sample.get("pred_eval", ""),
            comment=sample.get("comment", ""),
        )
        for sample in eval_set
    ]


def dict_to_sql_query(query: dict) -> SQLQuery:
    return SQLQuery(
        query=query["query"],
        results=(
            None
            if query["results"] is None
            else [tuple(row) for row in query["results"]]
        ),
    )


def save_eval(eval_set: list[SQLSample], path: Path) -> None:
    with path.open("w") as f:
        dump([asdict(sample) for sample in eval_set], f)


def eval_pipeline(eval_set: list[SQLSample], fuzzy: bool = True):
    correct = 0
    failed = 0
    for sample in eval_set:
        if sample.prediction.results is None:
            failed += 1
            continue
        if sample.check(fuzzy=fuzzy):
            correct += 1
    print(f"{failed} queries failed")
    return correct / len(eval_set)


def normalise_dict(classes: dict) -> dict:
    total = sum(v for _, v in classes.items())
    return {k: v / total for k, v in classes.items()}


def get_columns(sql_query: SQLQuery) -> set[tuple[str, str]]:
    query = sql_query.query
    parsed = parse_one(query, read="sqlite")
    columns = parsed.find_all(exp.Column)
    columns = set((column.this.this, column.table) for column in columns)
    return columns


def evaluate_sample(sample: SQLSample) -> str:
    if sample.prediction.results is None or len(sample.prediction.results) == 0:
        return "definitely_wrong"
    if sample.check(fuzzy=True) or sample.pred_eval == "y":
        return "definitely_correct"
    if sample.pred_eval == "m":
        return "possibly_correct"
    if sample.pred_eval == "n":
        return "possibly_wrong"

    prediction_columns = get_columns(sample.prediction)
    for label in sample.labels:
        correct_columns = get_columns(label)
        if prediction_columns.issuperset(correct_columns):
            return "possibly_correct"
    return "possibly_wrong"


def correct_upper_bound(eval_set: list[SQLSample]):
    classes = defaultdict(lambda: 0)
    for sample in eval_set:
        classes[evaluate_sample(sample)] += 1
    return normalise_dict(classes)
