import os
import json
import matplotlib.pyplot as plt
from argparse import ArgumentParser

import pandas as pd
import seaborn as sns
sns.set()

if __name__ == "__main__":
    green = "#76B900"
    gray = "#8C8C8C"
    blue = "#0071C5"
    red = "#890C58"

    parser = ArgumentParser()
    parser.add_argument("-i",  type=str, required=True)
    parser.add_argument("-o",  type=str, required=True)
    parser.add_argument("--trt-stats",  type=str, required=False, default=None)

    args = parser.parse_args()
    gpu_name = os.path.dirname(args.i)

    with open(args.i, 'r') as f:
        stats_ort = json.load(f)
    ep = stats_ort.pop("eps")
    dml_stats = {}
    cuda_stats = {}
    trt_stats = {}
    for k, v in stats_ort.items():
        dml_stats[k] = v["dml"]
        cuda_stats[k] = v["cuda"]
        trt_stats[k] = v["tensorrt"]

    trt_stats = pd.DataFrame(trt_stats)
    trt_stats.name = "trt"
    # overwrite setup time with engine deserialization time
    if args.trt_stats is not None:
        with open(args.trt_stats, 'r') as f:
            trt_raw_stats = json.load(f)
        trt_version = trt_raw_stats.pop("versions")['tensorrt']
        additionals = trt_raw_stats.pop("additional_args")
        deserialize_time = pd.DataFrame(trt_raw_stats).T['deserialization_time_ms']
        trt_stats.loc["creation_time_ms"] = deserialize_time

    dml_stats = pd.DataFrame(dml_stats)
    dml_stats.name = "dml"
    cuda_stats = pd.DataFrame(cuda_stats)
    cuda_stats.name = "cuda"

    dataframes = [trt_stats, cuda_stats, dml_stats]
    df = pd.concat([d for d in dataframes], axis=0,
                   keys=[d.name for d in dataframes])
    df.to_csv(os.path.join(args.o, "summary.csv"))
    df = df.T
    print(df)

    kwargs = {
        "figsize": (10, 6),
        "rot": 45,
    }

    def index_df(k):
        temp = df.iloc[:, df.columns.get_level_values(1) == k]
        temp.columns = temp.columns.get_level_values(0)
        return temp

    def index(
            k, dataframe): return dataframe.iloc[:, dataframe.columns.get_level_values(1) == k]

    plot = index_df("mean_end_to_end_latency_ms").plot.bar(
        color=[green, gray, blue],  ylabel="ms", **kwargs)
    plot.set_title("Inference latency {}".format(gpu_name))
    plt.tight_layout()
    plot.figure.savefig(os.path.join(args.o, 'inference_ort.png'))

    plot = index_df("creation_time_ms").plot.bar(
        color=[green, gray, blue], ylabel="ms", **kwargs)
    trt_add = " (TRT reports deserialization time)" if args.trt_stats is not None else ""
    plot.set_title("ORT context creation time {}".format(gpu_name))
    plt.tight_layout()
    plot.figure.savefig(os.path.join(args.o, 'creation_time.png'))
