"""gittables
"""
import os
import os.path as osp
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from datetime import datetime as dt


def study_one_dataset(
    idx: int, df_path: str, df_name: str, group_name: str, info_folder: str
):
    info_dict = {
        "group_name": group_name,
        "name": df_name,
        "tot_rows": 0,
        "tot_columns": 0,
        "tot_values": 0,
        "num_columns": 0,
        "cat_columns": 0,
        "uniq_cat_values": 0,
        "avg_red_cat_values": 0,
    }

    # if osp.exists(osp.join(info_folder, group_name) + ".csv"):
    #     return (idx, None)

    try:
        df = pd.read_parquet(df_path)

        info_dict["tot_rows"], info_dict["tot_columns"] = df.shape
        info_dict["tot_values"] = df.shape[0] * df.shape[1]
        num_attr = df.select_dtypes(include="number").columns
        cat_attr = [_ for _ in df.columns if _ not in num_attr]
        info_dict["num_columns"] = len(num_attr)
        info_dict["cat_columns"] = len(cat_attr)

        val_str = df[cat_attr].values.astype(str).ravel()

        uniq_values, count_uniq = np.unique(val_str, return_counts=True)
        info_dict["uniq_cat_values"] = len(uniq_values)
        if len(uniq_values) > 0:
            info_dict["avg_red_cat_values"] = np.mean(count_uniq)

        return (idx, info_dict)
    except Exception:
        return (idx, None)


if __name__ == "__main__":

    root_dir = osp.realpath("data/zenodo/")

    tables_dir = osp.join(root_dir, "tables")
    num_dir = len(os.listdir(tables_dir))

    start_time = dt.now()
    print(f"Start time: {start_time}")

    for folder_name in tqdm(
        os.listdir(tables_dir), total=num_dir, position=1, leave=True
    ):
        tgt_folder = osp.join(tables_dir, folder_name)
        assert osp.exists(tgt_folder)
        info_folder = osp.join(root_dir, "info_tables")
        assert osp.exists(info_folder)
        tot_files = len(os.listdir(tgt_folder))

        if tot_files == 0:
            print(f"Folder {tgt_folder} is empty, skipping.")
            continue

        data_dict = {}
        columns = [
            "group_name",
            "name",
            "tot_rows",
            "tot_columns",
            "tot_values",
            "num_columns",
            "cat_columns",
            "uniq_cat_values",
            "avg_red_cat_values",
        ]

        r = Parallel(n_jobs=8, verbose=0)(
            delayed(study_one_dataset)(
                idx, osp.join(tgt_folder, fname), fname, folder_name, info_folder
            )
            for idx, fname in tqdm(
                enumerate(os.listdir(tgt_folder)),
                position=0,
                leave=False,
                total=tot_files,
            )
        )
        data_dict = {k: v for (k, v) in r if v is not None}

        info_df = pd.DataFrame.from_dict(data_dict, orient="index", columns=columns)
        info_df.to_csv(osp.join(info_folder, folder_name) + "_stats.csv")

    end_time = dt.now()
    print(f"End time: {end_time}")
    print(f"Time required: {(end_time - start_time).total_seconds():}")
