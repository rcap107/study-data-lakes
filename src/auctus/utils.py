import pandas as pd
import polars as pl
from pathlib import Path
import json
from typing import Union
from src.utils import read_table_csv

def get_base_paths(dataset_path: Path, ds_name: str):
    """Given the dataset path and the dataset name, generate the required paths
    assuming the D3M directory tree structure. 

    Args:
        dataset_path (Path): Path to the dataset.
        ds_name (str): Name of the dataset.

    Returns:
        (Path, Path, Path): tuple of paths.
    """
    seed_path = Path(dataset_path, f"{ds_name}_dataset")
    problem_path = Path(dataset_path, f"{ds_name}_problem")
    cand_path = Path(dataset_path, f"{ds_name}_candidates")
    return seed_path,problem_path, cand_path


def get_unique_keys(left_table, left_on):
    uk = find_unique_keys(left_table, left_on)
    if uk is not None:
        return len(uk)
    else:
        return None


def find_unique_keys(df, key_cols, engine="polars"):
    """Find the set of unique keys given a combination of columns. 
    
    This function is used to find what is the potential cardinality of a join key.

    Args:
        df (Union[pd.DataFrame, pl.DataFrame]): Dataframe to estimate key cardinality on.
        key_cols (list): List of key columns.
        engine (str, optional): Either `polars` or `pandas`, the engine to be used. Defaults to "polars".

    Raises:
        ValueError: Raised if the engine is different from either `pandas` or `polars`.

    Returns:
        _type_: List of unique keys.
    """
    if engine == "pandas":
        unique_keys = df[key_cols].groupby(key_cols).size()
    elif engine == "polars":
        try:
            unique_keys = df.select(pl.col(key_cols)).groupby(key_cols).count()
        except pl.DuplicateError:
            unique_keys = None
    else:
        raise ValueError(f"Unknown engine {engine}")
    
    return unique_keys

def prepare_base_dict_info():
    d_info = {
        "ds_name": None,
        "candidate_name": None,
        "left_on": None,
        "right_on": None,
        "merged_rows": None,
        "scale_factor": None,
        "left_unique_keys": None,
        "right_unique_keys": None
    }
    return d_info
    

def merge_table(left_table: Union[pd.DataFrame, pl.DataFrame], 
                right_table: Union[pd.DataFrame, pl.DataFrame], 
                left_on: list, right_on: list, 
                engine: str ="polars", 
                how: str ="left"):
    """Merge tables according to the specified engine. 

    Args:
        left_table (Union[pd.DataFrame, pl.DataFrame]): Left table to be joined.
        right_table (Union[pd.DataFrame, pl.DataFrame]): Right table to be joined.
        left_on (list): Join keys in the left table.
        right_on (list): Join keys in the right table.
        engine (str, optional): Engine to be used. Defaults to "polars".
        how (str, optional): Join type. Defaults to "left".

    Raises:
        ValueError: Raises ValueError if the engine provided is not in [`polars`, `pandas`].

    Returns:
        Union[pd.DataFrame, pl.DataFrame]: Merged table.
    """
    if (type(left_table) == pl.DataFrame) and (type(right_table) == pl.DataFrame):
        merged = left_table[left_on].lazy().join(
        right_table[right_on], left_on=left_on, 
        right_on=right_on, how=how)
        return merged.collect()
    elif (type(left_table) == pd.DataFrame) and (type(right_table) == pd.DataFrame):
        merged = left_table[left_on].merge(
        right_table[right_on], left_on=left_on, 
        right_on=right_on, how=how)
        return merged
    else:
        raise TypeError(f"Incorrect dataframe type.")    


def profile_dataset(dataset_path: Path, engine="polars", verbose=0):
    """This function takes as input the path to a csv file, reads it using either 
    polars or pands, then runs a series of profiling operation to study the 
    join candidates provided by Auctus. 
    
    It notes down `ds_name`, `candidate_name`, `left_on`, `right_on`, size of the
    left join in `merged_rows`, `scale_factor`, cardinality of left and right 
    key columns in `left_unique_keys` and `right_unique_keys`. 

    Args:
        dataset_path (Path): Path to the dataset (assumes d3m+auctus format.)
        engine (str, optional): DataFrame engine to be used, either `polars` or `pandas`. Defaults to "polars".
        verbose (int, optional): How much information on dataset failures to be printed. Defaults to 0.

    Returns:
        pd.DataFrame: Summary of the statistics in DataFrame format.
    """
    ds_name = dataset_path.stem

    seed_path, problem_path, cand_path = get_base_paths(dataset_path, ds_name)
    
    left_table_path = Path(seed_path, "tables/learningData.csv")

    metadata_path = Path(cand_path, "queryResults.json")
    if metadata_path.exists() and left_table_path.exists():
        left_table = read_table_csv(left_table_path, engine)
        metadata = json.load(open(metadata_path))
        all_info_dict = {}
        idx_info = 0

        candidate_dict = metadata["candidate_datasets"]
        for cand_id, cand_info in candidate_dict.items():
            right_table_path = Path(cand_path, cand_id, "tables/learningData.csv")
            base_dict_info = dict(prepare_base_dict_info())
            base_dict_info["ds_name"] = ds_name
            base_dict_info["candidate_name"] = cand_id
            if not right_table_path.exists():
                if verbose>0:
                    print(f"{right_table_path} not found.")
                dict_info = dict(base_dict_info)
                all_info_dict[idx_info] = dict_info
                idx_info += 1
            else:
                right_table = read_table_csv(right_table_path, engine)
                for join_cand in cand_info["metadata"]:
                    left_on = join_cand["left_columns_names"][0]
                    right_on = join_cand["right_columns_names"][0]
                    dict_info = dict(base_dict_info)
                    if len(right_on) != len(left_on):
                        if verbose>1:
                            print(f"Left: {left_on} != Right: {right_on}")
                    elif any(r not in right_table.columns for r in right_on) or any(l not in left_table.columns for l in left_on):
                        if verbose>0:
                            print(f"Not all columns found.")
                    else:
                        dict_info["left_rows"], dict_info["left_cols"] = left_table.shape
                        
                        merged = merge_table(
                            left_table=left_table, 
                            right_table=right_table,
                            left_on=left_on,
                            right_on=right_on,
                            engine=engine
                        )
                        
                        dict_info["left_on"] = left_on
                        dict_info["right_on"] = right_on
                        dict_info["merged_rows"] = len(merged)
                        dict_info["scale_factor"] = len(merged) / dict_info["left_rows"]
                        dict_info["left_unique_keys"] = get_unique_keys(left_table, left_on)
                        dict_info["right_unique_keys"] = get_unique_keys(right_table, right_on)
                        
                    all_info_dict[idx_info] = dict_info
                    idx_info += 1
        df_info = pd.DataFrame().from_dict(all_info_dict, orient="index")
    else:
        df_info = pd.DataFrame()
    return df_info

