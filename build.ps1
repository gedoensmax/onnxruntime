$trt_rtx_1_1 = "A:\binaries\libraries\tensorrt\TensorRT-RTX-1.1.1.26"
$trt_rtx_1_2 = "A:\binaries\libraries\tensorrt\TensorRT-RTX-1.2.0.44"
$cuda = $env:CUDA_PATH
$config = "Debug"

python "tools\ci_build\build.py" --build_dir build_trt_1_1 --build --update --use_cache --config $config --parallel `
    --cuda_home $env:CUDA_PATH `
    --use_nv_tensorrt_rtx --build_shared_lib --build_wheel --tensorrt_rtx_home $trt_rtx_1_1 --cmake_generator "Ninja" --cmake_extra_defines CMAKE_EXPORT_COMPILE_COMMANDS=ON onnxruntime_BUILD_CPP_SAMPLES=ON

python "tools\ci_build\build.py" --build_dir build_trt_1_2 --build --update --use_cache --config $config --parallel `
    --cuda_home $env:CUDA_PATH `
    --use_nv_tensorrt_rtx --build_shared_lib --build_wheel --tensorrt_rtx_home $trt_rtx_1_2 --cmake_generator "Ninja" --cmake_extra_defines CMAKE_EXPORT_COMPILE_COMMANDS=ON onnxruntime_BUILD_CPP_SAMPLES=ON
