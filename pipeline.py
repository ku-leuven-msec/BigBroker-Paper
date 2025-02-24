import gc
import sys 
import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

from time import gmtime, strftime, time
from multiprocessing import set_start_method
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

from clustering import find_clusters, generate_cluster_helper_files
from social_graph import social_linking as social_linking_cpu
from social_graph_gpu import social_linking as social_linking_gpu


if __name__ == "__main__":
    set_start_method("spawn")

    functions = [
        find_clusters,
        generate_cluster_helper_files,
        #social_linking_cpu,
        social_linking_gpu
    ]

    versions = [
        "BE",
        #"NL",
        #"ES",
        #"FR1",
        #"FR2"
    ]

    configs = [{
        "working_dir": f"./run/{version}/5",
        "dataset_file": f"./datasets/{version}.parquet",
        "clustering": {
            "eps_spatial": 5,
            #"num_cores": 1
        },
        "social_linking": {
            "distances": [15.0, 5.0],
            #"num_cores": 1
        }
    } for version in versions]

    configs += [{
        "working_dir": f"./run/{version}/15",
        "dataset_file": f"./datasets/{version}.parquet",
        "clustering": {
            "eps_spatial": 15,
            #"num_cores": 1
        },
        "social_linking": {
            "distances": [25.0, 15.0],
            #"num_cores": 1
        }
    } for version in versions]

    with ProcessPoolExecutor(max_workers=1, max_tasks_per_child=1) as pool:
        for c in tqdm(configs, total=len(configs), desc="Run"):
            config = c

            print(f"Started processing {config['working_dir']} at {strftime('%Y-%m-%d %H:%M:%S', gmtime())}")
            start = time()

            if not os.path.exists(config["working_dir"]):
                os.makedirs(config["working_dir"])

            for fn in tqdm(functions, total=len(functions), desc="Stage"):
                start2 = time()
                print(f"Started {fn.__name__} at {strftime('%Y-%m-%d %H:%M:%S', gmtime())}")
                pool.submit(fn,config).result()
                end2 = time()
                print(f"Finished {fn.__name__} at {strftime('%Y-%m-%d %H:%M:%S', gmtime())} and took {end2-start2} seconds")

            end = time()
            print(f"Finished processing {config['working_dir']} at {strftime('%Y-%m-%d %H:%M:%S', gmtime())} and took {end-start} seconds")
