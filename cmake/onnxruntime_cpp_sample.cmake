# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

file(GLOB_RECURSE onnxruntime_sample_srcs CONFIGURE_DEPENDS
        "${REPO_ROOT}/samples/cpp/*.cc"
)

if (NOT onnxruntime_BUILD_SHARED_LIB)
    message(FATAL_ERROR "Samples can only be built with the shared lib enabled: onnxruntime_BUILD_SHARED_LIB=ON")
endif ()

onnxruntime_add_executable(onnxruntime_cpp_sample ${onnxruntime_sample_srcs})
target_include_directories(onnxruntime_cpp_sample PUBLIC "${REPO_ROOT}/include/onnxruntime")
if (UNIX)
    target_compile_options(onnxruntime_cpp_sample PUBLIC "-Wno-error=comment")
    target_compile_options(onnxruntime_cpp_sample PUBLIC "-Wno-error=unused")
    target_compile_options(onnxruntime_cpp_sample PUBLIC "-Wno-error=uninitialized")
else ()
    target_compile_options(onnxruntime_cpp_sample PRIVATE /wd4100)
    target_compile_options(onnxruntime_cpp_sample PRIVATE /wd4101)
    target_compile_options(onnxruntime_cpp_sample PRIVATE /wd4189)
    target_compile_options(onnxruntime_cpp_sample PRIVATE /wd4700)
endif ()
set_target_properties(onnxruntime_cpp_sample PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(onnxruntime_cpp_sample PROPERTIES FOLDER "ONNXRuntime")


target_link_libraries(onnxruntime_cpp_sample PRIVATE onnxruntime)

