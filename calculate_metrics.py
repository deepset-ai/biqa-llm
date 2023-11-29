import sys
from pathlib import Path

from eval import correct_upper_bound, load_eval_json

input_path = Path(sys.argv[1])

eval_set = load_eval_json(input_path)

eval_dict = correct_upper_bound(eval_set)
print(eval_dict)
print(
    f"Total correct: {eval_dict['definitely_correct'] + eval_dict['possibly_correct']:.3f}"
)
