import sys
from pathlib import Path

import pandas as pd

from eval import load_eval_json


def main():
    eval_set = load_eval_json(Path(sys.argv[1]))
    if len(sys.argv) == 3 and sys.argv[2] == "-c":
        for i, e in enumerate(eval_set):
            if e.comment != "":
                print(f"{i} | {e.comment} | {e.pred_eval}")
        sys.exit()

    if len(sys.argv) == 3 and sys.argv[2] == "-q":
        for i, e in enumerate(eval_set):
            print(f"{i} | {e.question}")
        sys.exit()

    index = 0

    while True:
        example = eval_set[index]

        for i, label in enumerate(example.labels, start=1):
            print(f"Correct Query #{i}:\n{label.query}")
            print(f"Correct Results #{i}:")
            print(pd.DataFrame(label.results).head(30))
        if example.prediction is not None:
            print(f"\nPredicted Query:\n{example.prediction.query}")
            print("Predicted Results:")
            print(pd.DataFrame(example.prediction.results).head(30))
        print(f"{index + 1}/{len(eval_set)}: {example.question}")
        print(f"Eval: {example.pred_eval}")
        print(f"Comment: {example.comment}")
        command = input(f"\n~>(n/q/<int>[0,{len(eval_set) - 1}]): ")
        if command == "q":
            break
        if command == "n":
            index = (index + 1) % len(eval_set)
        else:
            try:
                index = int(command)
            except ValueError:
                pass


if __name__ == "__main__":
    main()
