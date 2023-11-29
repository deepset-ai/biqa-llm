# %%
from pathlib import Path

from config import COLUMN_DESCRIPTION_FILE, DB_FILE
from eval import load_eval_json
from retrieval import Retriever
from sql_common import SQLExecutor

sql_executor = SQLExecutor(DB_FILE, COLUMN_DESCRIPTION_FILE)

eval_set = load_eval_json(Path("data/eval_set_multi_answers_res.json"))
# %%
tables, _ = sql_executor.execute("SELECT name FROM sqlite_schema WHERE type ='table' AND name NOT LIKE 'sqlite_%';")
tables = [t[0] for t in tables]
len(tables)
# %%
def find_tables_from_query(tables, query):
    query_tables = []

    for t in tables:
        if t.lower() in query.lower():
            query_tables.append(t)

    return set(query_tables)
# %%
# Just to see if multiple answer instances had different or same tables
mult_same = {'y': 0, 'n': 0}
for e in eval_set:
    if len(e.labels) > 1:
        t1 = find_tables_from_query(tables, e.labels[0].query)
        t2 = find_tables_from_query(tables, e.labels[1].query)
        if t1 == t2:
            mult_same['y'] += 1
        else:
            print(t1)
            print(t2)
            mult_same['n'] += 1
mult_same
# %%
def compute_recalls(retriever, tables, eval_set, per_col=False):
    recalls = []
    for e in eval_set:
        retrived_tables = retriever.run(e.question)['docs']

        for tk in range(1, len(tables)):
            max_recall = 0.
            if per_col:
                # One table can appear multiple times
                # We go on until the num of unique tables is tk
                retrieved_tk = []
                retrieved_register = {}
                for retrieved_col_table in retrived_tables:
                    if len(retrieved_tk) >= tk:
                        break
                    if retrieved_col_table not in retrieved_register:
                        retrieved_register[retrieved_col_table] = True
                        retrieved_tk.append(retrieved_col_table)
                retrieved_tk = set(retrieved_tk)
            else:
                retrieved_tk = set(retrived_tables[:tk])

            for label in e.labels:
                label_tables = find_tables_from_query(tables, label.query)
                if len(label_tables) == 0:
                    print(e)
                recall = len(retrieved_tk.intersection(label_tables)) / len(label_tables)
                if recall > max_recall:
                    max_recall = recall
            recalls.append((tk, max_recall))
    return recalls
# %%
# Simulate a perfect retriever 
class PerfectRetriever:
    def __init__(self, tables, eval_set) -> None:
        self.gt_mapping = {
            e.question : {'docs': list(find_tables_from_query(tables, e.labels[0].query))}
            for e in eval_set
        }

    def run(self, question):
        return self.gt_mapping[question]

recalls_gt = compute_recalls(PerfectRetriever(tables, eval_set), tables, eval_set)
# %%
retriever = Retriever(tables,
                      "sentence-transformers/all-mpnet-base-v2",
                      list(sql_executor.get_table_reprs(num_examples=3).values()),
                      top_k=len(tables))

recalls = compute_recalls(retriever, tables, eval_set)

retriever_desc = Retriever(tables,
                      "sentence-transformers/all-mpnet-base-v2",
                      list(sql_executor.get_table_reprs(num_distinct=10).values()),
                      top_k=len(tables))

recalls_desc = compute_recalls(retriever_desc, tables, eval_set)
# %% 
import pandas as pd

col_df = pd.read_csv(COLUMN_DESCRIPTION_FILE)
retriever_pc = Retriever(
    col_df['Table'].tolist(),
    "sentence-transformers/all-mpnet-base-v2",
    col_df['Description'].tolist(),
    top_k=len(col_df)
) 
recalls_pc = compute_recalls(retriever_pc, tables, eval_set, per_col=True)
# %%
import pandas as pd

df_gt = pd.DataFrame(recalls_gt, columns=['TopK', 'Recall'])
df_gt_mean = df_gt.groupby(by='TopK')['Recall'].mean()
df = pd.DataFrame(recalls, columns=['TopK', 'Recall'])
df_mean = df.groupby(by='TopK')['Recall'].mean()
dfd = pd.DataFrame(recalls_desc, columns=['TopK', 'Recall'])
dfd_mean = dfd.groupby(by='TopK')['Recall'].mean()
df_pc = pd.DataFrame(recalls_pc, columns=['TopK', 'Recall'])
df_pc_mean = df_pc.groupby(by='TopK')['Recall'].mean()
# %%
import matplotlib.pyplot as plt

plt.style.use("ggplot")

df_gt_mean.plot(label='Perfect Retriever')
df_mean.plot(label='Desc + 3 examples')
dfd_mean.plot(label='Desc + 10 distinct items')
df_pc_mean.plot(label='Per column retrieval')
plt.ylabel("Recall")
plt.legend()
plt.title("Table Retrieval: Recall vs TopK")
plt.show()
# %%
