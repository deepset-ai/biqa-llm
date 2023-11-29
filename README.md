## BIQA with LLMs

Explorations in using latest LLMs for BIQA.

### Data

The data is from the [Stack Overflow Developer Survey 2023](https://survey.stackoverflow.co/2023/).

- `data/eval_set_multi_answers_res.json`: Question and query pairs as list of `SQLSample`s with possibly more than one valid SQL for a question. Also results included.
- `data/survey_results_normalized_v2.db`: The main sqlite file. Download from here: [deepset/stackoverflow-survey-2023-text-sql](https://huggingface.co/datasets/deepset/stackoverflow-survey-2023-text-sql).

Or download as:
```console
wget -O data/survey_results_normalized_v2.db "https://drive.google.com/uc?export=download&id=1e_knoK9rYgWe8ADUw3PC8Fp6Jhnjgoms&confirm=t"
```

### Environment setup

```console
pip install -r requirements.txt
```

### Running the Evaluation

_Note: To use the OpenAI models, the `OPENAI_API_KEY` environment variable needs to be set. Can also put in `.env` file to be loaded by `python-dotenv`_

Create annotations i.e. fill in the predictions for the eval set:

```console
python create_annotations.py \
       -i data/eval_set_multi_answers_res.json \
       -o eval_preds.json \
```

Evaluate manually i.e. go through each evaluation (where pred available) and label them correct or not. If labelled correct, the prediction is added to the labels/answers.
```console
python evaluate_manually.py eval_preds.json eval_preds_manual.json
```

Calculate the final performance metrics:

```console
python calculate_metrics.py eval_preds_manual.json
```

### Different Approaches

Schema + Examples:
```console
python create_annotations.py \
       -i data/eval_set_multi_answers_res.json \
       -o eval_preds.json \
```

Schema + raw description
```console
python create_annotations.py \
 -i data/eval_set_multi_answers_res.json\
 -o eval_preds_base_raw_desc.json \
 -g base --raw-description \
```

Schema + column descriptions + few shot
```console
python create_annotations.py \
-i data/eval_set_multi_answers_res.json \
-o eval_preds_base_col_desc_fs.json \
-g base -d per-column --few-shot
```

Agents:
```console
python create_annotations.py \
 -i data/eval_set_multi_answers_res.json \
 -o eval_preds_agents.json \
 -g agent
```

(Perfect) Retrieval:
```console
python create_annotations.py \ 
 -i data/eval_set_multi_answers_res.json \
 -o eval_preds_retriever.json \
 -g retriever -d per-column \
```

### Running the API

For the application to work with OpenAI models, the `OPENAI_API_KEY` environment variable needs to be set.

You can set it directly or put it in the `.env` file.

```console
uvicorn api.main:app --reload --host="0.0.0.0" --port=8000
```

### Helper scripts

`column_descriptions.py`: To generate the "seed" column-level description file (to be completed manually)
`count_tokens.py`: Count number of tokens or to view the final prompt
`retriever_analysis.py`: Analysis of the retriever performance + plot generation
`view_eval_set.py`: View the data in the eval (or prediction) set

### Appendix

#### Data Creation

Created with this [Notebook](https://colab.research.google.com/drive/12NUeRMsld0toXMSXKFMaQVAv58XwOAT1?usp=sharing); uses [this spreadsheet](https://docs.google.com/spreadsheets/d/1Xh_TgMbyitvtw08g0byEmBpkwDGZDdBYenthOzcK6qI/edit?usp=sharing) defining manual adjustments.