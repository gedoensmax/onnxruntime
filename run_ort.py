import os
from argparse import ArgumentParser
import subprocess
import json
import re
import subprocess


def main(args):
    # -d [cudnn_conv_algorithm]:
    #       Specify CUDNN convolution algothrithms: 0(benchmark), 1(heuristic), 2(default).
    # -o [optimization level]: Default is 99 (all). Valid values are 0 (disable), 1 (basic), 2 (extended), 99 (all).
    ort_exec = args.exe
    ep_opts = {}
    if args.dml:
        ep_opts[("dml", "dml")] = []
    if args.trt:
        ep_opts[("tensorrt", "tensorrt")] = []
    if args.cuda:
        ep_opts[("cuda", "cuda")] = ["-q", "-d", "1"]
    if args.cuda_nhwc:
        ep_opts[("cuda", "cuda_nhwc")] = ["-q", "-d", "1", "-i", "prefer_nhwc|1"]

    # if directory run all models inside
    inputs = args.i
    if len(inputs) == 1:
        if os.path.isdir(inputs[0]):
            inputs = [
                os.path.join(inputs[0], p)
                for p in os.listdir(inputs[0])
                if p.endswith(".onnx")
            ]

    pathes = [os.path.abspath(p) for p in inputs]
    for p in pathes:
        if not os.path.exists(p):
            RuntimeError(f"Path does not exist {p}")

    results = {}
    for p in pathes:
        name = os.path.basename(p).split(".")[0]
        print(f"Running model: {name}")

        results[name] = {}
        for ep_name, opt in ep_opts.items():
            ep, config_name = ep_name
            print(config_name)
            command = [ort_exec, "-I", "-t", str(args.t), "-e", ep, *opt, p]
            results[name][config_name] = {"command": " ".join(command)}
            print(results[name][config_name]["command"])
            try:
                output = subprocess.check_output(
                    command,
                    stderr=subprocess.STDOUT,
                )
                output = output.decode("utf-8")
            except subprocess.CalledProcessError as e:
                print(
                    "command '{}' return with error (code {}): {}".format(
                        " ".join(e.cmd), e.returncode, e.output
                    )
                )
                results[name][config_name]["success"] = False
                continue

            results[name][config_name].update({
                "mean_end_to_end_latency_ms": float(
                    re.findall(r"Average inference time cost: ([\d.]*) ms", output)[0]
                ),
                "min_end_to_end_latency_ms": float(
                    re.findall(r"Min Latency: ([\d.]*) s", output)[0]
                )
                                             * 1000,
                "max_end_to_end_latency_ms": float(
                    re.findall(r"Max Latency: ([\d.]*) s", output)[0]
                )
                                             * 1000,
                "deserialization_time_ms": float(
                    re.findall(r"Session creation time cost: ([\d.]*) s", output)[0]
                )
                                           * 1000,
                "success": True
            })

    if args.o != "stdout":
        json_object = json.dumps(results, indent=4)
        with open(os.path.abspath(args.o), "w") as outfile:
            outfile.write(json_object)
    else:
        print(results)


if __name__ == "__main__":
    parser = ArgumentParser(
        "ONNX runtime perf test",
        description="""Can run a folder or a list of model provided via `-i`. 
        Inference is timed over `-t` seconds 
        and results are printed to stdout if there is no json filespecified ie `-o`.
        Per default only DML is used but CUDA EP or TRT ep can be enabled.
        To alter EP setting please modify the code.""",
    )
    parser.add_argument("-i", nargs="+", type=str, required=True)
    parser.add_argument("-o", type=str, default="stdout")
    parser.add_argument(
        "-t", type=int, default=10, help="Duration to benchmark the network in seconds"
    )
    parser.add_argument("--dml", action="store_true", help="DML EP")
    parser.add_argument("--trt", action="store_true", help="TRT EP")
    parser.add_argument("--cuda", action="store_true", help="CUDA EP")
    parser.add_argument("--cuda_nhwc", action="store_true", help="CUDA EP NHWC")
    parser.add_argument(
        "--exe",
        type=str,
        default=".\\onnxruntime_perf_test.exe",
        help="Path to onnxruntime_perf_test executable",
    )
    args = parser.parse_args()
    main(args)
