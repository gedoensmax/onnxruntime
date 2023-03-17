---
title: Profiling tools 
grand_parent: Performance
parent: Tune performance
nav_order: 1
---

# Profiling Tools

## Contents
{: .no_toc }

* TOC placeholder
{:toc}


## ONNX Go Live Tool (OLive)

The [ONNX Go Live "OLive" tool](https://github.com/microsoft/OLive) is a Python package that automates the process of accelerating models with ONNX Runtime. It contains two parts: (1) model conversion to ONNX with correctness validation (2) auto performance tuning with ORT. Users can run these two together through a single pipeline or run them independently as needed.

As a quickstart, please see the [notebook tutorials](https://github.com/microsoft/OLive/tree/master/notebook-tutorial) and [command line examples](https://github.com/microsoft/OLive/tree/master/cmd-example) 

## In-code performance profiling

The onnxruntime_perf_test.exe tool (available from the build drop) can be used to test various knobs. Please find the usage instructions using `onnxruntime_perf_test.exe -h`. The [perf_view tool](https://github.com/microsoft/onnxruntime/tree/main/tools/perf_view) can also be used to render the statistics as a summarized view in the browser.

You can enable ONNX Runtime latency profiling in code:

```python
import onnxruntime as rt

sess_options = rt.SessionOptions()
sess_options.enable_profiling = True
```

If you are using the onnxruntime_perf_test.exe tool, you can add `-p [profile_file]` to enable performance profiling.

In both cases, you will get a JSON file which contains the detailed performance data (threading, latency of each operator, etc). This file is a standard performance tracing file, and to view it in a user-friendly way, you can open it by using chrome://tracing:

* Open Chrome browser
* Type chrome://tracing in the address bar
* Load the generated JSON file

To profile CUDA kernels, please add the cupti library to your PATH and use the onnxruntime binary built from source with `--enable_cuda_profiling`.
To profile ROCm kernels, please add the roctracer library to your PATH and use the onnxruntime binary built from source with `--enable_rocm_profiling`. 

Performance numbers from the device will then be attached to those from the host. For example:

```json
{"cat":"Node", "name":"Add_1234", "dur":17, ...}
{"cat":"Kernel", "name":"ort_add_cuda_kernel", dur:33, ...}
```

Here, the "Add" operator from the host initiated a CUDA kernel on device named "ort_add_cuda_kernel" which lasted for 33 microseconds.
If an operator called multiple kernels during execution, the performance numbers of those kernels will all be listed following the call sequence:

```json
{"cat":"Node", "name":<name of the node>, ...}
{"cat":"Kernel", "name":<name of the kernel called first>, ...}
{"cat":"Kernel", "name":<name of the kernel called next>, ...}
```