import pandas as pd
import polars as pl
from typing import Union

def read_table_csv(table_path, engine="polars"):
    """Read a single table using different either polars or pandas as dataframe engine.

    Args:
        table_path (Path): Path to the table to be loaded
        engine (str, optional): DataFrame engine to be used, either `polars` or `pandas`. Defaults to "polars".

    Raises:
        ValueError: The provided engine is unknown.

    Returns:
        _type_: Dataframe read according to the required engine. 
    """
    if engine == "polars":
        return pl.read_csv(table_path, infer_schema_length=0)
    elif engine == "pandas":
        return pd.read_csv(table_path, low_memory=False).astype(str)
    else:
        raise ValueError(f"Unknown engine {engine}")



def col_operations_pd(column: pd.Series):
    col_stats = {
                "count_unique": column.nunique(),
                "count_nans": column.isna().sum(),
                "nan_frac": column.isna().sum()/len(column)        
                 }
    
    return pd.Series(col_stats)


def profile_pandas(df: pd.DataFrame):
    profiling_dict = {}
    for col in df.columns:
        profiling_dict[col] = col_operations_pd(df[col])
    
    return pd.DataFrame(profiling_dict)


def col_operations_pl(column: pl.Series):
    col_stats = {
                "count_unique": column.n_unique(),
                "count_nans": column.null_count(),
                "nan_frac": column.null_count()/len(column)        
                 }
    
    return col_stats

def profile_polars(df: pl.DataFrame):
    profiling_dict = {}
    for col in df.columns:
        profiling_dict[col] = col_operations_pl(df[col])
    
    return pd.DataFrame(profiling_dict)

def profile_dataframe(df: Union[pd.DataFrame, pl.DataFrame]):
    if type(df) == pd.DataFrame:
        pass
    elif type(df) == pl.DataFrame:
        return profile_polars(df)
    else:
        raise TypeError(f"Table {df} has inappropriate type {type(df)}")
