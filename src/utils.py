import pandas as pd
import polars as pl
from pathlib import Path
import timeit

def read_df_pl(df_path: Path):
    return pl.read_csv(df_path)

def read_df_pd(df_path: Path):
    return pd.read_csv(df_path)

def read_df_pl_to_pd(df_path: Path):
    return pl.read_csv(df_path).to_pandas()

