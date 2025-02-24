import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import json
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from numba import njit
from multiprocessing import cpu_count, Pool, set_start_method

pd.set_option("mode.copy_on_write", True)
pd.options.mode.copy_on_write = True

df: pd.DataFrame
offsets: np.ndarray
social_distances: list
num_cores : int = max(cpu_count() - 1, 1)


def social_linking_init(sc, data):
    global df, offsets, social_distances
    social_distances = sc
    df, offsets = data


@njit
def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    Reference:
        https://stackoverflow.com/a/29546836/7657658
    """
    #lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    lon1 = np.radians(lon1)
    lon2 = np.radians(lon2)
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c
    return km


def social_linking_worker(idx):
    global df, offsets, social_distances
    row = df.iloc[idx]

    df2 = df.iloc[idx + 1:offsets[idx]]

    if len(df2) == 0:
        return []

    distances = 1000 * haversine_np(df2["LON"].values, df2["LAT"].values, row["LON"], row["LAT"])

    results = []
    for social_dist in social_distances:
        filtered_df = df2.iloc[np.flatnonzero(distances < social_dist)]

        if len(filtered_df) == 0:
            break

        matches = []
        for record in filtered_df.itertuples():
            if df.iloc[idx, 0] != record.DEVICE_ID:
                matches.append([
                    df.iloc[idx, 0],
                    record.DEVICE_ID,
                    max(df.iloc[idx, 3], record.EVENT_START) / (24 * 3600),
                    min(df.iloc[idx, 4], record.EVENT_END) / (24 * 3600),
                ])
        results.append(matches)

    return results


def social_linking(config):
    global df, offsets, social_distances, num_cores
    working_dir = config["working_dir"]
    social_distances = list(sorted(config["social_linking"]["distances"], key=lambda x: -x))
    if "num_cores" in config["social_linking"]:
        num_cores = config["social_linking"]["num_cores"]

    df = pd.read_parquet(
        f"{working_dir}/clusters_temporal_minimized.parquet",
        columns=["DEVICE_ID", "LAT", "LON", "EVENT_START", "EVENT_END", "CLUSTERS_TEMPORAL"])

    df = df[df["CLUSTERS_TEMPORAL"] > -1]
    df.drop(columns=["CLUSTERS_TEMPORAL"])
    df.reset_index(drop=True, inplace=True)
    df[["EVENT_START", "EVENT_END"]] = np.round(df[["EVENT_START", "EVENT_END"]] * 24 * 3600).astype(np.int32)

    offsets = df["EVENT_START"].searchsorted(df["EVENT_END"], 'right')
    df_size = len(df)

    device_hashes = df["DEVICE_ID"].unique()
    mapping = [{int(device_hash): {} for device_hash in device_hashes} for _ in social_distances]

    with Pool(num_cores, initializer=social_linking_init, initargs=(social_distances, (df, offsets),)) as pool:
        for social_match in tqdm(pool.imap(social_linking_worker, range(df_size)),
                                 total=df_size,
                                 desc="Socially linking"):
            if len(social_match):
                for idx, tuples in enumerate(social_match):
                    tmp = mapping[idx]
                    for entry in tuples:
                        (personA, personB, start_time, end_time) = entry
                        if personB not in tmp[personA]:
                            tmp[personA][personB] = []
                        tmp[personA][personB].append((start_time, end_time))

    for mapping_entry in mapping:
        for key, value in list(mapping_entry.items()):
            if not value:
                del mapping_entry[key]

    for idx, social_dist in enumerate(social_distances):
        with open(f'{working_dir}/social_graph_{social_dist}.json', 'w') as f:
            json.dump(mapping[idx], f)


if __name__ == '__main__':
    set_start_method("spawn")

    config = {
        "working_dir": f"./test",
        "social_linking": {
            "distances": [15.0, 5.0],
            # "num_cores": 1
        }
    }

    social_linking(config)
