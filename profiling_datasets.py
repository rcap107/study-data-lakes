"""d3m+auctus
This script is used to run the profiling operation on all datasets in the 
given data_folder with an engine passed as argument. 
"""

import pandas as pd 
import polars as pl
import json
from pathlib import Path
from src.auctus.utils import profile_dataset
from tqdm import tqdm
import sys

# Setting working folder
working_folder = Path(".")
root_data_folder = Path(working_folder, "data")
data_folder = Path(root_data_folder, "soda-data-lake/a-d3m/a-d3m_full")
assert data_folder.exists()

dataset_list = list(data_folder.iterdir())

df_info_overall = pd.DataFrame()

for dataset_path in tqdm(dataset_list, total=len(dataset_list)):
    #TODO: Something explodes with a-d3m_full on 174, possibly running out of 
    #TODO: memory. 
    df_info = profile_dataset(dataset_path=dataset_path, engine=sys.argv[1])
    df_info_overall = pd.concat([df_info_overall, df_info])

df_info_overall.to_csv("info_a-d3m_full.csv", index=False)
