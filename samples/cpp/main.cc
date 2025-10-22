#include <d3d12.h>
#include <dxgi1_6.h>
#include <filesystem>
#include <iostream>
#include <chrono>

#include <onnxruntime_cxx_api.h>
#include <onnxruntime_run_options_config_keys.h>
#include <onnxruntime_session_options_config_keys.h>
#include <onnxruntime/core/providers/nv_tensorrt_rtx/nv_provider_options.h>
#include <thread>
#include <wrl.h>

// CUDA includes
#include <cuda_d3d11_interop.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include "nv/target"
//#include <NvInfer.h>
//#define NVTX
#ifdef NVTX
#include <nvtx3/nvtx3.hpp>
#else
namespace nvtx3
{
    struct scoped_range
    {
        scoped_range(std::string)
        {
        }
    };
}
#endif


inline void error_check(cudaError_t err, const char* file, int line)
{
    if (err != cudaSuccess)
    {
        ::fprintf(stderr, "CUDA ERROR at %s[%d] : %s\n", file, line,
                  cudaGetErrorString(err));
        abort();
    }
}

#define CUDA_CHECK(err)                                                        \
  do {                                                                         \
    error_check(err, __FILE__, __LINE__);                                      \
  } while (0)


size_t ONNXDtypeToBytes(const ONNXTensorElementDataType element_type)
{
    switch (element_type)
    {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: // maps to c type float
        return sizeof(float);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: // maps to c type uint8_t
        return sizeof(uint8_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: // maps to c type int8_t
        return sizeof(int8_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: // maps to c type uint16_t
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        return sizeof(uint16_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: // maps to c type int16_t
        return sizeof(int16_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: // maps to c type int32_t
        return sizeof(int32_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: // maps to c type int64_t
        return sizeof(int64_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
        return sizeof(bool);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: // maps to c type double
        return sizeof(double);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32: // maps to c type uint32_t
        return sizeof(uint32_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64: // maps to c type uint64_t
        return sizeof(uint64_t);
    default:
        std::cerr << "unexpected input data type" << std::endl;
        return 0;
    }
}

size_t ORTValueToBytes(const Ort::Value& value)
{
    const size_t bytes =
        value.GetTensorTypeAndShapeInfo().GetElementCount() *
        ONNXDtypeToBytes(value.GetTensorTypeAndShapeInfo().GetElementType());
    return bytes;
}


void describe_session(Ort::Session& session)
{
    Ort::AllocatorWithDefaultOptions cpu_alloc;
    {
        auto input_count = session.GetInputCount();
        std::cout << "Input count: " << input_count << std::endl;
        for (int input_idx = 0; input_idx < input_count; ++input_idx)
        {
            auto input_name = session.GetInputNameAllocated(input_idx, cpu_alloc);
            auto input_info = session.GetInputTypeInfo(input_idx);
            auto input_shape_info = input_info.GetTensorTypeAndShapeInfo();
            auto input_dtype = input_shape_info.GetElementType();
            std::cout << "\tinput index: " << input_idx << std::endl;
            std::cout << "\t  input dtype: " << input_dtype << std::endl;
            std::cout << "\t  input shape: ";
            auto shape = input_shape_info.GetShape();
            for (auto& s : shape)
            {
                std::cout << s << ", ";
            }
            std::cout << std::endl;
        }
    }
    {
        auto output_count = session.GetOutputCount();
        std::cout << "output count: " << output_count << std::endl;
        for (int output_idx = 0; output_idx < output_count; ++output_idx)
        {
            auto output_name = session.GetOutputNameAllocated(output_idx, cpu_alloc);
            auto output_info = session.GetOutputTypeInfo(output_idx);
            auto output_shape_info = output_info.GetTensorTypeAndShapeInfo();
            auto output_dtype = output_shape_info.GetElementType();
            std::cout << "\toutput index: " << output_idx << std::endl;
            std::cout << "\t  output dtype: " << output_dtype << std::endl;
            std::cout << "\t  output shape: ";
            auto shape = output_shape_info.GetShape();
            for (auto& s : shape)
            {
                std::cout << s << ", ";
            }
            std::cout << std::endl;
        }
    }
}

void run_with_gpu_bindings(Ort::Session& session, int iterations,
                           cudaStream_t stream)
{
    cudaStream_t upload_stream, download_stream;
    CUDA_CHECK(cudaStreamCreate(&download_stream));
    CUDA_CHECK(cudaStreamCreate(&upload_stream));
    int device_id = 0;
    Ort::MemoryInfo memory_info_cuda_pinned("CudaPinned",
                                            OrtAllocatorType::OrtArenaAllocator,
                                            device_id, OrtMemTypeDefault);
    Ort::Allocator cpu_alloc(session, memory_info_cuda_pinned);

    Ort::MemoryInfo memory_info_cuda("Cuda", OrtAllocatorType::OrtArenaAllocator,
                                     device_id, OrtMemTypeDefault);
    Ort::Allocator gpu_alloc(session, memory_info_cuda);

    OrtAllocator* allocator = gpu_alloc;
    Ort::IoBinding io_binding(session);
    std::vector<Ort::Value> output_values_flop;
    std::vector<Ort::Value> output_values_flip;
    std::vector<std::string> output_names;
    std::vector<const char*> output_names_raw;
    {
        auto output_count = session.GetOutputCount();
        for (int output_idx = 0; output_idx < output_count; ++output_idx)
        {
            auto output_name = session.GetOutputNameAllocated(output_idx, cpu_alloc);
            auto output_info = session.GetOutputTypeInfo(output_idx);
            auto output_shape_info = output_info.GetTensorTypeAndShapeInfo();
            auto output_dtype = output_shape_info.GetElementType();
            auto shape = output_shape_info.GetShape();
            output_values_flop.emplace_back(Ort::Value::CreateTensor(
                allocator, shape.data(), shape.size(), output_dtype));
            output_values_flip.emplace_back(Ort::Value::CreateTensor(
                allocator, shape.data(), shape.size(), output_dtype));
            io_binding.BindOutput(output_name.get(), output_values_flop.back());
            output_names.emplace_back(std::string{output_name.get()});
            output_names_raw.emplace_back(output_names.back().c_str());
        }
    }

    CUDA_CHECK(cudaProfilerStart());

    std::vector<Ort::Value> input_values;
    std::vector<std::string> input_names;
    std::vector<const char*> input_names_raw;
    {
        auto input_count = session.GetInputCount();
        for (int input_idx = 0; input_idx < input_count; ++input_idx)
        {
            auto input_name = session.GetInputNameAllocated(input_idx, cpu_alloc);
            auto input_info = session.GetInputTypeInfo(input_idx);
            auto input_shape_info = input_info.GetTensorTypeAndShapeInfo();
            auto input_dtype = input_shape_info.GetElementType();
            auto shape = input_shape_info.GetShape();
            input_values.emplace_back(Ort::Value::CreateTensor(
                allocator, shape.data(), shape.size(), input_dtype));
            io_binding.BindInput(input_name.get(), input_values.back());
            input_names.push_back(std::string{input_name.get()});
            input_names_raw.emplace_back(input_names.back().c_str());
        }
    }
    std::vector<Ort::Value> input_values_cpu;
    std::vector<Ort::Value> output_values_cpu;
    {
        auto input_count = session.GetInputCount();
        for (int input_idx = 0; input_idx < input_count; ++input_idx)
        {
            auto input_name = session.GetInputNameAllocated(input_idx, cpu_alloc);
            auto input_info = session.GetInputTypeInfo(input_idx);
            auto input_shape_info = input_info.GetTensorTypeAndShapeInfo();
            auto input_dtype = input_shape_info.GetElementType();
            auto shape = input_shape_info.GetShape();
            input_values_cpu.emplace_back(Ort::Value::CreateTensor(
                cpu_alloc, shape.data(), shape.size(), input_dtype));
        }
    }
    {
        auto output_count = session.GetOutputCount();
        for (int output_idx = 0; output_idx < output_count; ++output_idx)
        {
            auto output_name = session.GetOutputNameAllocated(output_idx, cpu_alloc);
            auto output_info = session.GetOutputTypeInfo(output_idx);
            auto output_shape_info = output_info.GetTensorTypeAndShapeInfo();
            auto output_dtype = output_shape_info.GetElementType();
            auto shape = output_shape_info.GetShape();
            output_values_cpu.emplace_back(Ort::Value::CreateTensor(
                cpu_alloc, shape.data(), shape.size(), output_dtype));
        }
    }
    {
        cudaEvent_t inference_done, inference_done_cpu, upload_done;
        CUDA_CHECK(cudaEventCreate(&inference_done));
        CUDA_CHECK(cudaEventCreate(&inference_done_cpu));
        CUDA_CHECK(cudaEventCreate(&upload_done));

        // set to done to not block first run
        CUDA_CHECK(cudaEventRecord(inference_done, stream));

        nvtx3::scoped_range async{"async gpu tensors"};
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i)
        {
            const std::vector<Ort::Value>& output_values =
                static_cast<bool>(i % 2) ? output_values_flip : output_values_flop;
            const std::string output_iter =
                static_cast<bool>(i % 2) ? "flip" : "flop";
            {
                nvtx3::scoped_range async{"upload"};
                for (int out_idx = 0; out_idx < output_values_cpu.size(); ++out_idx)
                {
                    const size_t bytes = ORTValueToBytes(input_values[out_idx]);
                    CUDA_CHECK(
                        cudaMemcpyAsync(input_values[out_idx].GetTensorMutableRawData(),
                            input_values_cpu[out_idx].GetTensorRawData(),
                            bytes, cudaMemcpyHostToDevice, upload_stream));
                }
                CUDA_CHECK(cudaEventRecord(upload_done, upload_stream));
            }
            for (int idx_out = 0; idx_out < output_names.size(); ++idx_out)
            {
                io_binding.BindOutput(output_names[idx_out].c_str(),
                                      output_values[idx_out]);
            }
            CUDA_CHECK(
                cudaStreamWaitEvent(stream, upload_done)); // should be almost a no-op
            {
                nvtx3::scoped_range async{"inference"};
                Ort::RunOptions run_options;
                run_options.AddConfigEntry("disable_synchronize_execution_providers",
                                           "1");
                session.Run(run_options, io_binding);
                cudaEventRecord(inference_done, stream);
            }
            CUDA_CHECK(cudaEventSynchronize(
                inference_done_cpu)); // ensure that we do not sen too many commands
            // to the GPU
            CUDA_CHECK(cudaEventRecord(inference_done_cpu, stream));
            {
                nvtx3::scoped_range async{"download " + output_iter};
                CUDA_CHECK(cudaStreamWaitEvent(download_stream, inference_done));
                for (int out_idx = 0; out_idx < output_values_cpu.size(); ++out_idx)
                {
                    const size_t bytes = ORTValueToBytes(output_values_flop[out_idx]);
                    CUDA_CHECK(cudaMemcpyAsync(
                        output_values_cpu[out_idx].GetTensorMutableRawData(),
                        output_values[out_idx].GetTensorRawData(), bytes,
                        cudaMemcpyDeviceToHost, download_stream));
                }
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

        std::cout << "GPU bindings per iteration [ms]: "
            << static_cast<float>(duration.count()) /
            static_cast<float>(iterations) / 1000.f
            << std::endl;
    }
    CUDA_CHECK(cudaProfilerStop());
}

void run_with_cpu_bindings(Ort::Session& session, int iterations)
{
    int device_id = 0;
    Ort::AllocatorWithDefaultOptions cpu_alloc;
    Ort::MemoryInfo memory_info_cuda("Cuda", OrtAllocatorType::OrtArenaAllocator,
                                     device_id, OrtMemTypeDefault);
    Ort::Allocator gpu_alloc(session, memory_info_cuda);
    OrtAllocator* allocator = cpu_alloc;

    Ort::IoBinding io_binding(session);
    std::vector<Ort::Value> output_values_flop;
    std::vector<Ort::Value> output_values_flip;
    std::vector<std::string> output_names;
    std::vector<const char*> output_names_raw;
    {
        auto output_count = session.GetOutputCount();
        for (int output_idx = 0; output_idx < output_count; ++output_idx)
        {
            auto output_name = session.GetOutputNameAllocated(output_idx, cpu_alloc);
            auto output_info = session.GetOutputTypeInfo(output_idx);
            auto output_shape_info = output_info.GetTensorTypeAndShapeInfo();
            auto output_dtype = output_shape_info.GetElementType();
            auto shape = output_shape_info.GetShape();
            output_values_flop.emplace_back(Ort::Value::CreateTensor(
                allocator, shape.data(), shape.size(), output_dtype));
            output_values_flip.emplace_back(Ort::Value::CreateTensor(
                allocator, shape.data(), shape.size(), output_dtype));
            io_binding.BindOutput(output_name.get(), output_values_flop.back());
            output_names.emplace_back(std::string{output_name.get()});
            output_names_raw.emplace_back(output_names.back().c_str());
        }
    }

    CUDA_CHECK(cudaProfilerStart());
    {
        nvtx3::scoped_range sync{"inputs and outputs on CPU"};
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i)
        {
            std::vector<Ort::Value> input_values;
            std::vector<std::string> input_names;
            std::vector<const char*> input_names_raw;
            {
                auto input_count = session.GetInputCount();
                for (int input_idx = 0; input_idx < input_count; ++input_idx)
                {
                    auto input_name = session.GetInputNameAllocated(input_idx, cpu_alloc);
                    auto input_info = session.GetInputTypeInfo(input_idx);
                    auto input_shape_info = input_info.GetTensorTypeAndShapeInfo();
                    auto input_dtype = input_shape_info.GetElementType();
                    auto shape = input_shape_info.GetShape();
                    input_values.emplace_back(Ort::Value::CreateTensor(
                        cpu_alloc, shape.data(), shape.size(), input_dtype));
                    io_binding.BindInput(input_name.get(), input_values.back());
                    input_names.push_back(std::string{input_name.get()});
                    input_names_raw.emplace_back(input_names.back().c_str());
                }
            }

            Ort::RunOptions run_options;
            session.Run(run_options, io_binding);
        }
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

        std::cout << "CPU bindings per iteration [ms]: "
            << static_cast<float>(duration.count()) /
            static_cast<float>(iterations) / 1000.f
            << std::endl;
    }

    CUDA_CHECK(cudaProfilerStop());
}

int main(int argc, char* argv[])
{
    static auto ort_env = Ort::Env(ORT_LOGGING_LEVEL_WARNING);
    static auto ort_api = Ort::GetApi();

    std::cout << "ORT API VERSION: " << ORT_API_VERSION << std::endl;

    Ort::SessionOptions session_options;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    std::string ctx_file = "ctx.onnx";
    const std::wstring widestr_ctx =
     std::wstring(ctx_file.begin(), ctx_file.end());
    const ORTCHAR_T* ctx_path_cstr = widestr_ctx.c_str();
    // if (std::filesystem::exists(ctx_file))
    // {
    //     std::filesystem::remove(ctx_file);
    // }
    try
    {
        // ort_api.AddFreeDimensionOverrideByName(session_options, "2B", 2);
        // ort_api.AddFreeDimensionOverrideByName(session_options, "H", 64);
        // ort_api.AddFreeDimensionOverrideByName(session_options, "W", 64);
        std::string model_path("resnet101-v2-7_bs8.onnx");

        if (argc > 1)
        {
            model_path = argv[1];
        }
        const std::wstring widestr =
            std::wstring(model_path.begin(), model_path.end());
        const ORTCHAR_T* model_path_cstr = widestr.c_str();

        int device_id = 0;
        Ort::SessionOptions session_options;
        // session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
        // session_options.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
        // session_options.AddConfigEntry(kOrtSessionOptionEpContextFilePath, ctx_file.c_str());
        session_options.AppendExecutionProvider("NvTensorRTRTXExecutionProvider",
                                                {
                                                    {onnxruntime::nv::provider_option_names::kDeviceId, std::to_string(device_id)},
                                                    {onnxruntime::nv::provider_option_names::kRuntimeCacheFile, "rt_cache/"}
                                                });
        if (!std::filesystem::exists(ctx_file)) {
          Ort::ModelCompilationOptions compile_options(ort_env, session_options);
          compile_options.SetInputModelPath(model_path_cstr);
          compile_options.SetOutputModelPath(ctx_path_cstr);
          Ort::CompileModel(ort_env, compile_options);
          std::cout << "model compiled" << std::endl;
        } else {
          std::cout << "model ctx present" << std::endl;
        }
        Ort::Session session(ort_env, ctx_path_cstr, session_options);
        std::cout << "model constructed" << std::endl;
        if (!session) {
          throw std::exception("Failed to create session");
        }
        describe_session(session);
        run_with_cpu_bindings(session, 10); // warmup run
        int iterations = 100;
        run_with_gpu_bindings(session, iterations, stream);
        //    run_with_cpu_bindings(session, iterations);
    }
    catch (std::runtime_error& e)
    {
        std::cerr << e.what() << std::endl;
    }
    catch (Ort::Exception& e)
    {
      std::cerr << "ORT Exception: " <<  e.what() << std::endl;
    }
    std::cout << "DONE" << std::endl;
    return 0;
}
