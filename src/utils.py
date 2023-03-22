import pandas as pd
import polars as pl
from pathlib import Path
import timeit
import json

def read_df_pl(df_path: Path):
    return pl.read_csv(df_path)

def read_df_pd(df_path: Path):
    return pd.read_csv(df_path)

def read_df_pl_to_pd(df_path: Path):
    return pl.read_csv(df_path).to_pandas()


def find_unique_keys_pandas(df: pd.DataFrame, key_cols):
    return df[key_cols].groupby(key_cols).size()
def find_unique_keys_polars(df: pl.DataFrame, key_cols):
    return df.select(pl.col(key_cols)).groupby(key_cols).count()
def find_unique_keys_polars_lazy(df: pl.LazyFrame, key_cols):
    return df.select(pl.col(key_cols)).groupby(key_cols).count().collect()

def find_unique_keys(df, key_cols, engine="polars"):
    if engine == "pandas":
        unique_keys = find_unique_keys_pandas(df, key_cols)
    elif engine == "polars":
        try:
            unique_keys = find_unique_keys_polars(df, key_cols)
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


def read_table(table_path, engine="polars"):
    if engine == "polars":
        return pl.read_csv(table_path, infer_schema_length=0)
    elif engine == "pandas":
        return pd.read_csv(table_path, low_memory=False).astype(str)
    else:
        raise ValueError(f"Unknown engine {engine}")
    

def merge_table(left_table, right_table, left_on, right_on, engine):
    if engine == "polars":
        merged = left_table[left_on].join(
        right_table[right_on], left_on=left_on, 
        right_on=right_on, how="left")
        return merged
    elif engine == "pandas":
        merged = left_table[left_on].merge(
        right_table[right_on], left_on=left_on, 
        right_on=right_on, how="left")
        return merged
    else:
        raise ValueError(f"Unknown engine {engine}")    

def get_unique_keys(left_table, left_on):
    uk = find_unique_keys(left_table, left_on)
    if uk is not None:
        return len(uk)
    else:
        return None

def profile_dataset(dataset_path: Path, engine="polars", verbose=False):
    ds_name = dataset_path.stem

    seed_path = Path(dataset_path, f"{ds_name}_dataset")
    problem_path = Path(dataset_path, f"{ds_name}_problem")
    cand_path = Path(dataset_path, f"{ds_name}_candidates")
    
    path_left = Path(seed_path, "tables/learningData.csv")
    assert path_left.exists()
    left_table = read_table(path_left, engine)
    
    
    metadata_path = Path(cand_path, "queryResults.json")
    if metadata_path.exists():
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
                if verbose:
                    print(f"{right_table_path} not found.")
                dict_info = dict(base_dict_info)
                all_info_dict[idx_info] = dict_info
                idx_info += 1
            else:
                right_table = read_table(right_table_path, engine)
                for join_cand in cand_info["metadata"]:
                    left_on = join_cand["left_columns_names"][0]
                    right_on = join_cand["right_columns_names"][0]
                    dict_info = dict(base_dict_info)
                    if len(right_on) != len(left_on):
                        if verbose:
                            print(f"Left: {left_on} != Right: {right_on}")
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
