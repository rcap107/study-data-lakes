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

@pl.api.register_expr_namespace("old_profiling")
class ProfilingOperations:
    """From https://stackoverflow.com/a/75892816/3741342
    """
    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr
        
    def n_unique(self):
        return self._expr.n_unique().suffix("___count_unique")
    
    def null_count(self):
        return self._expr.null_count().suffix("___count_nans")
    
    def nan_frac(self):
        return (self._expr.null_count()/self._expr.count()).suffix("___nan_frac")

    def count(self):
        return self._expr.count().suffix("___count")
    

@pl.api.register_expr_namespace("profiling")
class Profiling:
    """From https://stackoverflow.com/a/75892816/3741342
    """
    def __init__(self, expr):
        self._expr = expr
    def n_unique(self):
        return pl.struct(
            [
                pl.lit("count_unique").alias("stat"), 
                self._expr.n_unique().cast(pl.Float64())
            ]
        )
    def null_count(self):
        return pl.struct(
            [
                pl.lit("count_nans").alias("stat"), 
                self._expr.null_count().cast(pl.Float64())
            ]
        )
    def nan_frac(self):
        return pl.struct(
            [
                pl.lit("nan_frac").alias("stat"), 
                (self._expr.null_count()/self._expr.count()).cast(pl.Float64())
            ]
        )

    def count(self):
        return pl.struct(
            [
                pl.lit("count").alias("stat"), 
                (self._expr.count()).cast(pl.Float64())
            ]
        )

def profile_with_col_ops_v1(df: pl.DataFrame):
    q=(df.lazy().select(
        getattr(
            pl.all().old_profiling, x
        )() for x in dir(pl.Expr.old_profiling) if x[:1]!="_"
    ).melt().with_columns(
        varsplit=pl.col("variable").str.split("___")
    ).with_columns(
        c1=pl.col("varsplit").arr.first(),
        c2=pl.col("varsplit").arr.last()
    ).collect().pivot("value", "c2", "c1"))

    return q.to_pandas()

def profile_with_col_ops(df: pl.DataFrame):
    df = df.lazy()
    q = (pl.concat(
        [
            df.select(
                getattr(pl.all().profiling, stat_method)()
            ).unnest('stat') 
            for stat_method in dir(pl.Expr.profiling) 
            if stat_method[:1]!="_"
        ]
    ).collect())
    return q.to_pandas()

def profile_with_col_ops_v2(df: pl.DataFrame):
    q=df.lazy().select(
        getattr(
            pl.all().old_profiling, x
        )() for x in dir(pl.Expr.old_profiling) if x[:1]!="_"
    ).melt().with_columns(
        varsplit=pl.col("variable").str.split("___")
    ).with_columns(
        c1=pl.col("varsplit").arr.first(),
        c2=pl.col("varsplit").arr.last()
    ).collect()

    return q

