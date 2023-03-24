import json
import os
import os.path as osp
import pickle
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm


def extract_from_json(json_dict):
    info_dict = {}
    info_dict["problemName"] = json_dict["about"]["problemName"]

    info_dict["taskKeywords"] = json_dict["about"]["taskKeywords"]
    info_dict["performanceMetrics"] = [_["metric"] for _ in json_dict["inputs"]["performanceMetrics"]]

    return info_dict


def plot_stats(df_data, df_tasks, df_metrics):
    fig, ax = plt.subplots(1)
    sns.barplot(df_tasks.reset_index(), x="index", y="count")
    _ = plt.xticks(rotation=45, ha="right")
    plt.xlabel("Task")
    plt.tight_layout()
    fig.savefig("images/tasks.png")

    fig, ax = plt.subplots(1)
    sns.barplot(df_metrics.reset_index(), x="index", y="count", ax=ax)
    _ = plt.xticks(rotation=45, ha="right")
    plt.xlabel("Metric")
    plt.tight_layout()
    fig.savefig("images/metrics.png")

    fig, ax = plt.subplots(1)
    sns.scatterplot(data=df_data, x="rows", y="columns")
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Distribution of row/columns")
    plt.xlabel("Number of rows (log)")
    plt.ylabel("Number of columns (log)")
    plt.tight_layout()
    fig.savefig("images/dist-row-columns.png")


def parse_study_datasets_from_file(extract_from_json, v_datasets_path):
    counter_tasks = Counter()
    counter_metrics = Counter()
    dict_shapes = {}
    dict_df_metadata = {}

    with open(v_datasets_path, "r") as fp:
        n_rows = int(fp.readline().strip())
        for idx, row in tqdm(enumerate(fp), total=n_rows):
            folder = row.strip()
            dataset_name = osp.basename(folder)
            dataset_folder = dataset_name + "_dataset"
            problem_folder = dataset_name + "_problem"

            pth = osp.join(folder, dataset_folder)
            tables_folder = osp.join(pth, "tables")
            total_shape = np.array([0,0])
            for df_file in os.listdir(tables_folder):
                df = pd.read_csv(osp.join(tables_folder, df_file), low_memory=False)
                total_shape += df.shape

            json_metadata = json.load(open(
                osp.join(folder, problem_folder, "problemDoc.json")))
            info_metadata = extract_from_json(json_metadata)    

            counter_tasks.update(info_metadata["taskKeywords"])
            counter_metrics.update(info_metadata["performanceMetrics"])
            
            dict_shapes[dataset_name] = total_shape
            
            dict_df_metadata[idx] = {
                "name": dataset_name,
                "path": folder,
                "taskKeywords": tuple(info_metadata["taskKeywords"]),
                "performanceMetrics": tuple(info_metadata["performanceMetrics"]),
                "totalShape": total_shape
            }
        
    pickle.dump(dict_df_metadata, open("info_valid_datasets.json", "wb"))

    df_data = pd.DataFrame().from_dict(dict_shapes, orient="index", columns=["rows", "columns"])
    df_tasks = pd.DataFrame().from_dict(counter_tasks, orient="index", columns=["count"])
    df_metrics = pd.DataFrame().from_dict(counter_metrics, orient="index", columns=["count"])
    df_metadata = pd.DataFrame().from_dict(dict_df_metadata, orient="index")
    df_metadata["numCells"] = df_metadata["totalShape"].apply(lambda x: x[0]) * \
        df_metadata["totalShape"].apply(lambda x: x[1])
    return df_data,df_tasks,df_metrics,df_metadata

def extract_subset(df_metadata):
    small_set = {}
    for g, group in df_metadata.groupby("performanceMetrics"):
        selected = group.sort_values("numCells").iloc[len(group)//2]
        small_set.update({selected["path"]:g})

    return small_set


if __name__ == "__main__":
    v_datasets_path = "target_datasets.txt"

    df_data, df_tasks, df_metrics, df_metadata = parse_study_datasets_from_file(extract_from_json, v_datasets_path)

    dfs = dict(zip(["data", "tasks", "metrics", "metadata"], (df_data, df_tasks, df_metrics, df_metadata)))

    for name, df in dfs.items():
        df.to_csv(f"data/stats/stats_{name}.csv", index=False)

    small_set = extract_subset(df_metadata)

    with open("small_set.txt", "w") as fp:
        fp.write(f"{len(small_set)}\n")
        for pth in small_set.keys():
            fp.write(f"{pth}\n")
    
    # plot_stats(df_data, df_tasks, df_metrics)
    
