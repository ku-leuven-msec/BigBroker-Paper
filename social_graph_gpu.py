import os
from multiprocessing import set_start_method

from tqdm.auto import tqdm

import json
import numpy as np
import pandas as pd
from numba import cuda

from numba.cuda.libdevice import cosf, sqrtf, asinf, sinf


@cuda.jit(device=True)
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    Reference:
        https://stackoverflow.com/a/29546836/7657658
    """
    # lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sinf(dlat / 2.0) ** 2 + cosf(lat1) * cosf(lat2) * sinf(dlon / 2.0) ** 2

    c = 2 * asinf(sqrtf(a))
    km = 6371 * c
    return km


def social_linking(config):
    working_dir = config["working_dir"]
    social_distances = list(sorted(config["social_linking"]["distances"], key=lambda x: x))
    if "num_cores" in config["social_linking"]:
        num_cores = config["social_linking"]["num_cores"]

    df = pd.read_parquet(
        f"{working_dir}/clusters_temporal_minimized.parquet",
        columns=["DEVICE_ID", "LAT", "LON", "EVENT_START", "EVENT_END", "CLUSTERS_TEMPORAL"])

    df = df[df["CLUSTERS_TEMPORAL"] > -1]
    df.drop(columns=["CLUSTERS_TEMPORAL"])
    df.reset_index(drop=True, inplace=True)
    df[["EVENT_START", "EVENT_END"]] = np.round(df[["EVENT_START", "EVENT_END"]] * 24 * 3600).astype(np.int32)

    df[["LAT2", "LON2"]] = np.radians(df[["LAT", "LON"]])

    offsets = df["EVENT_START"].searchsorted(df["EVENT_END"], 'right')

    device_hashes = df["DEVICE_ID"].unique()
    mapping = [{int(device_hash): {} for device_hash in device_hashes} for _ in social_distances]

    event_starts = np.ascontiguousarray(df["EVENT_START"].values)
    d_event_starts = cuda.to_device(event_starts)

    event_ends = np.ascontiguousarray(df["EVENT_END"].values)
    d_event_ends = cuda.to_device(event_ends)

    latitudes = np.ascontiguousarray(df["LAT2"].values)
    d_latitudes = cuda.to_device(latitudes)

    longitudes = np.ascontiguousarray(df["LON2"].values)
    d_longitudes = cuda.to_device(longitudes)

    DEVICE_IDes = np.ascontiguousarray(df["DEVICE_ID"].values)
    d_DEVICE_IDes = cuda.to_device(DEVICE_IDes)

    offsets = np.ascontiguousarray(offsets)
    d_offsets = cuda.to_device(offsets)

    social_distances = np.array(social_distances, dtype=np.float32)

    @cuda.jit
    def social_linking_worker_gpu(indexes, res, index_offsets, time_offsets, event_starts, event_ends, latitudes,
                                  longitudes, DEVICE_IDes):
        pos = cuda.grid(1)
        if pos < len(indexes):  # Check array boundaries
            idx = indexes[pos]
            if index_offsets[idx] == 0:
                index_offsets[idx] = idx

            for index in range(index_offsets[idx] + 1, time_offsets[idx]):
                index_offsets[idx] = index
                distance = 1000 * haversine(longitudes[index], latitudes[index], longitudes[idx], latitudes[idx])
                for social_dist_idx, social_dist in enumerate(social_distances):
                    if distance < social_dist and DEVICE_IDes[index] != DEVICE_IDes[idx]:
                        res[pos, 0] = DEVICE_IDes[index]
                        res[pos, 1] = max(event_starts[idx], event_starts[index])
                        res[pos, 2] = min(event_ends[idx], event_ends[index])
                        res[pos, 3] = social_dist_idx
                        return

    indexes = np.array(range(len(df)), dtype=np.int32)
    d_index_offsets = cuda.device_array_like(indexes)

    with (tqdm(desc="Num iterations") as pbar):
        while len(indexes):
            d_indexes = cuda.to_device(indexes)
            res = np.zeros((len(indexes), 4), dtype=np.int32)
            d_res = cuda.to_device(res)

            social_linking_worker_gpu.forall(len(indexes))(d_indexes, d_res, d_index_offsets, d_offsets, d_event_starts,
                                                           d_event_ends, d_latitudes, d_longitudes, d_DEVICE_IDes)

            res = d_res.copy_to_host()

            indexes_with_contacts = np.flatnonzero(res[:, 0])

            if len(indexes_with_contacts):
                for index_with_contact in indexes_with_contacts:
                    idx = indexes[index_with_contact]
                    match = res[index_with_contact]
                    personA = int(DEVICE_IDes[idx])
                    personB = int(match[0])
                    start_time = match[1] / (24 * 3600)
                    end_time = match[2] / (24 * 3600)
                    for i in range(match[3], len(social_distances)):
                        tmp = mapping[i]
                        if personB not in tmp[personA]:
                            tmp[personA][personB] = []
                        tmp[personA][personB].append((start_time, end_time))

            indexes = np.ascontiguousarray(indexes[indexes_with_contacts], dtype=np.int32)
            pbar.update(1)

    for mapping_entry in mapping:
        for key, value in list(mapping_entry.items()):
            if not value:
                del mapping_entry[key]
            else:
                for key2 in mapping_entry[key]:
                    mapping_entry[key][key2].sort()

    for social_dist_idx, social_dist in enumerate(social_distances):
        with open(f'{working_dir}/social_graph_{social_dist}.json', 'w') as f:
            json.dump(mapping[social_dist_idx], f)


if __name__ == '__main__':
    set_start_method("spawn")

    config = {
        "working_dir": f"./test",
        "social_linking": {
            "distances": [15.0, 5.0],
        }
    }

    social_linking(config)
