"""Script to run manual evaluation.

Also allows for resuming eval on an output file.
"""
import sys
from pathlib import Path

import pandas as pd

from eval import load_eval_json, save_eval

input_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])

run_all = False
if len(sys.argv) > 3:
    if sys.argv[3] == "--all":
        run_all = True

eval_set_path = input_path

if output_path.exists():
    print(f"Output file: {output_path} exists.")
    reload = input("Do you want to reload and continue from it? (y/n): ")
    if reload == "y":
        eval_set_path = output_path

print(eval_set_path)
eval_set = load_eval_json(eval_set_path)

for j, example in enumerate(eval_set):
    if example.prediction.results is None:
        continue
    save_eval(eval_set, output_path)
    if run_all or (not example.check() and example.pred_eval == ""):
        for i, label in enumerate(example.labels, start=1):
            print(f"Correct Query #{i}:\n{label.query}")
            print(f"Correct Results #{i}:")
            print(pd.DataFrame(label.results).head(30))
        print(f"\nPredicted Query:\n{example.prediction.query}")
        print("Predicted Results:")
        print(pd.DataFrame(example.prediction.results).head(30))
        print(f"{j + 1}/{len(eval_set)}: {example.question}")
        response = None
        while response not in {"y", "n", "m"}:
            response = input("Is the prediction correct? (y/n/m): ")
            example.pred_eval = response
            example.comment = input("Comment: ")
        if response == "y":
            example.labels.append(example.prediction)

save_eval(eval_set, output_path)
