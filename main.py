from pathlib import Path
import timeit
from timerit import Timerit
from src.utils import read_df_pd, read_df_pl, read_df_pl_to_pd

working_folder = Path(".")
root_data_folder = Path(working_folder, "data")
data_folder = Path(root_data_folder, "soda-data-lake/a-d3m/a-d3m_dedup")
dataset_folder = Path(data_folder, "datasets")
assert dataset_folder.exists()

dataset_list = list(dataset_folder.iterdir())
df_path = dataset_list[0]

for timer in Timerit(num=200, verbose=2):
    with timer:
        read_df_pl(df_path)
for timer in Timerit(num=200, verbose=2):
    with timer:
        read_df_pd(df_path)
for timer in Timerit(num=200, verbose=2):
    with timer:
        read_df_pl_to_pd(df_path)

