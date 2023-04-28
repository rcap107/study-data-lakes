"""gittables
"""
import os
import os.path as osp
import pandas as pd
from pyarrow import parquet as pq
from tqdm import tqdm
import numpy as np
import json
from joblib import Parallel, delayed
from datetime import datetime as dt


def study_one_dataset(idx: int, tgt_folder: str, df_name: str, group_name: str):
    info_dict = {"group_name": group_name, "name": df_name}
    try:        
        file_path = osp.join(tgt_folder, df_name)
        table = pq.read_table(file_path)
        md = table.schema.metadata[b"gittables"]
        js = json.loads(md.decode("utf-8", errors="strict"))

        info_dict.update(js["table_domain"])
        return (idx, info_dict)
    except Exception as e:
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
            "schema_syntactic",
            "schema_semantic",
            "dbpedia_syntactic",
            "dbpedia_semantic",
        ]

        r = Parallel(n_jobs=8, verbose=0)(
            delayed(study_one_dataset)(idx, tgt_folder, fname, folder_name)
            for idx, fname in tqdm(
                enumerate(os.listdir(tgt_folder)),
                position=0,
                leave=False,
                total=tot_files,
            )
        )
        data_dict = {k: v for (k, v) in r if v is not None}

        info_df = pd.DataFrame.from_dict(data_dict, orient="index", columns=columns)
        info_df.to_csv(osp.join(info_folder, folder_name) + "_md.csv")

    end_time = dt.now()
    print(f"End time: {end_time}")
    print(f"Time required: {(end_time - start_time).total_seconds():}")
