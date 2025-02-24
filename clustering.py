import gc
import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

from sklearn.cluster import DBSCAN
from st_dbscan import ST_DBSCAN

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from multiprocessing import Pool, set_start_method, cpu_count
from concurrent.futures import ProcessPoolExecutor

pd.set_option("mode.copy_on_write", True)
pd.options.mode.copy_on_write = True

num_cores : int = cpu_count()
eps_spatial : float
eps_temporal : float = 1 / (24*2)
points_threshold = 50001
chunk_size = 50000


def find_clusters_worker_init(config):
    global eps_spatial, eps_temporal, points_threshold
    eps_spatial = config["clustering"]["eps_spatial"] * 0.001 / 6371.0088
    if "eps_temporal" in config["clustering"]:
        eps_temporal = config["clustering"]["eps_temporal"]
    if "points_threshold" in config["clustering"]:
        points_threshold = config["clustering"]["points_threshold"]


def find_clusters_worker(data):
    global eps_spatial, eps_temporal, points_threshold
    index, df = data

    df = df.copy()
    filter_duplicate = df.duplicated(keep=False)
    df.loc[filter_duplicate,["LAT", "LON"]] += np.random.choice(np.arange(-0.0000001, 0.0000001, 0.00000000001),
                                                                  sum(filter_duplicate) * 2).reshape((sum(filter_duplicate), 2)).astype(np.float32)
    radian_values = np.radians(df[["LAT", "LON"]].astype(np.float32))
    df["CLUSTERS_TEMPORAL"] = ST_DBSCAN(eps1=eps_spatial, eps2=eps_temporal, min_samples=3, n_jobs=1, metric_spatial='haversine').fit_frame_split_new(
        np.concatenate([df[["EVENT_TIMESTAMP"]], radian_values], axis=1), 200).labels.astype(np.int16)

    return df
    

def generate_chunks(config):
    dataset_file = config["dataset_file"]
    working_dir = config["working_dir"]

    df = pd.read_parquet(dataset_file,columns=['DEVICE_ID', 'LAT', 'LON', 'EVENT_TIMESTAMP'])
    index_new = np.lexsort((df["EVENT_TIMESTAMP"].values, df["DEVICE_ID"].values))
    df = df.reindex(index_new)
    df = df.reset_index()
    df["index"] = df["index"].astype(np.int32)
    df_grouping_index = np.unique(df["DEVICE_ID"], return_index=True)[1]

    offsets = [0] + [df_grouping_index[i] for i in range(chunk_size, len(df_grouping_index), chunk_size)] + [len(df)]

    for idx in tqdm(range(len(offsets) - 1), total=len(offsets) - 1, desc="Generating scratch files"):
        if os.path.exists(f"{working_dir}/clustered_scratch_{idx}.parquet"):
            continue
        df2 = df.iloc[offsets[idx]:offsets[idx + 1]]
        df2.to_parquet(f"{working_dir}/clustered_scratch_{idx}.parquet")

    n_chunks = idx + 1
    return n_chunks

def process_chunk(config, idx):
    global num_cores
    working_dir = config["working_dir"]
    df2 = pd.read_parquet(f"{working_dir}/clustered_scratch_{idx}.parquet")
    df_groups = list(df2.groupby("DEVICE_ID", as_index=False, observed=True))

    if "num_cores" in config["clustering"]:
        num_cores = config["clustering"]["num_cores"]

    del df2
    with Pool(num_cores, initializer=find_clusters_worker_init, initargs=(config,)) as pool:
        df2 = pd.concat(
            list(tqdm(pool.imap_unordered(find_clusters_worker, df_groups), total=len(df_groups),
                      desc="Clustering")), ignore_index=True)
    df2.to_parquet(f"{working_dir}/clustered_scratch_{idx}.parquet")
    del df2

def merge_chunks(config,n_chunks):
    working_dir = config["working_dir"]
    df = pd.concat(
        tqdm((pd.read_parquet(f"{working_dir}/clustered_scratch_{idx}.parquet") for idx in range(n_chunks)),
             total=n_chunks, desc="Merging scratch files"), ignore_index=True)

    for idx in range(n_chunks):
        os.remove(f"{working_dir}/clustered_scratch_{idx}.parquet")

    df.sort_values(by=["index"], inplace=True)
    df.drop(columns=["index"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.to_parquet(f"{working_dir}/clustered.parquet")


def find_clusters(config):

    with ProcessPoolExecutor(max_workers=1, max_tasks_per_child=1) as ppe:
        n_chunks = ppe.submit(generate_chunks, config).result()

        for idx in tqdm(range(n_chunks), total=n_chunks, desc="Processing scratch files"):
            ppe.submit(process_chunk,config,idx).result()

        ppe.submit(merge_chunks, config,n_chunks).result()


def generate_cluster_helper_files(config):
    working_dir = config["working_dir"]

    df = pd.read_parquet(f"{working_dir}/clustered.parquet")

    grouping = df.groupby(["DEVICE_ID", "CLUSTERS_TEMPORAL"])
    filtered_df = pd.concat([
        grouping["LAT"].mean(),
        grouping["LON"].mean(),
        grouping["EVENT_TIMESTAMP"].min().rename("EVENT_START"),
        grouping["EVENT_TIMESTAMP"].max().rename("EVENT_END"),
    ], axis=1).reset_index()

    filtered_df.sort_values(by=['EVENT_START'], ascending=True, inplace=True)
    filtered_df = filtered_df.reset_index(drop=True)

    filtered_df.to_parquet(f"{working_dir}/clusters_temporal_minimized.parquet")


if __name__ == '__main__':
    set_start_method("spawn")
    config = {
        "working_dir": f"./test",
        "dataset_file": f"./test/test.parquet",
        "clustering": {
            "eps_spatial" : 15,
            # "num_cores": 1
        }
    }

    find_clusters(config)
    generate_cluster_helper_files(config)
