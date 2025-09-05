// Copyright (c) Microsoft Corporation. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Licensed under the MIT License.
#include "core/framework/ortmemoryinfo.h"

#include <onnxruntime_cxx_api.h>
#include <core/providers/nv_tensorrt_rtx/nv_provider_options.h>
#include <onnxruntime_run_options_config_keys.h>
#include <onnxruntime_session_options_config_keys.h>
#include <core/graph/constants.h>
#include <vector>
#include <filesystem>
#include <iostream>
#include <chrono>
#include <codecvt>
#include <fstream>
#include <cuda_runtime.h>

#ifdef WIN32
#define STRING_CLASS std::wstring
#else
#define STRING_CLASS std::string
#endif

std::string PathToUTF8(const std::filesystem::path& path) {
#ifdef WIN32
  std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
  return converter.to_bytes(path);
#else
  return path.c_str();
#endif
}

std::vector<char> readBinaryFile(const std::string& filename) {
  // Open the stream for reading
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file: " + filename);
  }

  // Determine the size of the file
  file.seekg(0, std::ios::end);
  std::streamsize filesize = file.tellg();
  file.seekg(0, std::ios::beg);

  // Create a vector to hold the data
  std::vector<char> buffer(filesize);

  // Read the data from the stream into the vector
  if (!file.read(reinterpret_cast<char*>(buffer.data()), filesize)) {
    throw std::runtime_error("Could not read file: " + filename);
  }

  return buffer;
}

Ort::IoBinding generate_io_binding(Ort::Session& session, OrtAllocator* allocator = nullptr, std::unordered_map<std::string, std::vector<int64_t>> runtime_shape_override = {}) {
  Ort::IoBinding binding(session);
  auto default_allocator = Ort::AllocatorWithDefaultOptions();
  if (allocator == nullptr) {
    allocator = default_allocator;
  }
  const OrtMemoryInfo* info;
  Ort::ThrowOnError(Ort::GetApi().AllocatorGetInfo(allocator, &info));
  Ort::MemoryInfo mem_info(info->name, info->alloc_type, info->device.Id(), info->mem_type);
  for (int input_idx = 0; input_idx < int(session.GetInputCount()); ++input_idx) {
    auto input_name = session.GetInputNameAllocated(input_idx, Ort::AllocatorWithDefaultOptions());
    std::string input_name_str = input_name.get();
    auto full_tensor_info = session.GetInputTypeInfo(input_idx);
    auto tensor_info = full_tensor_info.GetTensorTypeAndShapeInfo();
    auto shape = tensor_info.GetShape();
    auto type = tensor_info.GetElementType();
    if (runtime_shape_override.find(input_name_str) != runtime_shape_override.end()) {
      shape = runtime_shape_override.at(input_name_str);
    }
    for (auto& v : shape) {
      if (v == -1) {
        v = 1;
      }
    }
    auto input_value = Ort::Value::CreateTensor(allocator,
                                                shape.data(),
                                                shape.size(),
                                                type);
    binding.BindInput(input_name.get(), input_value);
  }

  for (int output_idx = 0; output_idx < int(session.GetOutputCount()); ++output_idx) {
    auto output_name = session.GetOutputNameAllocated(output_idx, Ort::AllocatorWithDefaultOptions());
    binding.BindOutput(output_name.get(), mem_info);
  }
  return binding;
}

#define FILE_IO 1

int main() {
  // auto logging_level = OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR;
  // auto logging_level = OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING;
  auto logging_level = OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE;
  auto use_trt_rtx = true;
  auto weightless = true;
  auto regenerate = false;
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  char stream_str[32];
#ifdef WIN32
  sprintf_s(stream_str, "%llu", reinterpret_cast<uint64_t>(stream));
#else
  sprintf(stream_str, "%lu", reinterpret_cast<uint64_t>(stream));
#endif
  try {
    std::filesystem::path model_name = "refiner_fp32.onnx";
    std::filesystem::path external_file_name = "";
    std::unordered_map<std::string, std::vector<int64_t>> runtime_shape_override{};

    // std::filesystem::path model_name = "unet.onnx";
    // std::filesystem::path external_file_name = "";
    // std::unordered_map<std::string, std::vector<int64_t>> runtime_shape_override{
    //     {"sample", {2, 4, 16, 16}},
    //     {"timestep", {1}},
    //     {"encoder_hidden_states", {2, 77, 768}}};

    // std::filesystem::path model_name = "Phi-4-mini-instruct-INT4/model.onnx";
    // std::filesystem::path external_file_name = "Phi-4-mini-instruct-INT4/model.onnx_data";
    // int bs = 1, seq_q = 1, seq_max = 1024;
    // std::unordered_map<std::string, std::vector<int64_t>> runtime_shape_override{
    //     {"input_ids", {bs, seq_q}},
    //     {"attention_mask", {bs, seq_max}},
    // };
    // for (int i = 0; i < 32; ++i) {
    //   runtime_shape_override.insert({"past_key_values." + std::to_string(i) + ".key", {bs, 8, seq_max, 128}});
    //   runtime_shape_override.insert({"past_key_values." + std::to_string(i) + ".value", {bs, 8, seq_max, 128}});
    // }
#ifdef WIN32
    std::filesystem::path model_dir = "N:/models/";
#else
    std::filesystem::path model_dir = "/mnt/share/onnx/convnets/";
#endif
    std::filesystem::path model_file = model_dir / model_name;
    std::filesystem::path model_data_file = model_dir / external_file_name;

    if (!std::filesystem::exists(model_file)) {
      throw std::runtime_error("file does not exist");
    }

    std::filesystem::path model_ctx =
#if FILE_IO
        "model_file_io_ctx.onnx";
#else
        "model_ctx.onnx";
#endif
    auto env = Ort::Env();
    auto api = Ort::GetApi();
    env.UpdateEnvWithCustomLogLevel(logging_level);

    Ort::MemoryInfo mem_info_cuda("Cuda", OrtArenaAllocator, 0, OrtMemTypeDefault);

    // /*setup environment*/
    // {
    //   OrtArenaCfg* arena_cfg;
    //   std::array<const char*, 0> arena_option_name = {};
    //   std::array<const size_t, 0> arena_option_value = {};
    //
    //   std::unordered_map<std::string, std::string> provider_options;
    //   Ort::ThrowOnError(api.CreateArenaCfgV2(arena_option_name.data(),
    //                                          arena_option_value.data(),
    //                                          arena_option_name.size(),
    //                                          &arena_cfg));
    //
    //   env.CreateAndRegisterAllocatorV2(
    //       onnxruntime::kNvTensorRTRTXExecutionProvider,
    //       mem_info_cuda,
    //       provider_options,
    //       arena_cfg);
    // }

    // AOT time
    if (regenerate || !std::filesystem::exists(model_ctx)) {
      if (std::filesystem::exists(model_ctx)) {
        std::filesystem::remove(model_ctx);
      }
      auto start = std::chrono::high_resolution_clock::now();
      Ort::SessionOptions so;
      // so.SetGraphOptimizationLevel(ORT_DISABLE_ALL);
      // Ort::ThrowOnError(api.AddFreeDimensionOverrideByName(so, "batch_size", 1));
      // Ort::ThrowOnError(api.AddFreeDimensionOverrideByName(so, "sequence_length", 1));
      // Ort::ThrowOnError(api.AddFreeDimensionOverrideByName(so, "total_sequence_length", 1024));
      // Ort::ThrowOnError(api.AddFreeDimensionOverrideByName(so, "past_sequence_length", 1024));

      if (use_trt_rtx) {
        so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
        so.AddConfigEntry(kOrtSessionOptionEpContextEmbedMode, "1");
        so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, PathToUTF8(model_ctx).c_str());
        so.AppendExecutionProvider(onnxruntime::kNvTensorRTRTXExecutionProvider, {});
        // so.AppendExecutionProvider(onnxruntime::kNvTensorRTRTXExecutionProvider,
        //                            {{"nv_profile_min_shapes", "sample:1x4x8x8,timestep:1,encoder_hidden_states:2x77x768"},
        //                             {"nv_profile_opt_shapes", "sample:2x4x16x16,timestep:1,encoder_hidden_states:2x77x768"},
        //                             {"nv_profile_max_shapes", "sample:1024x4x10024x10024,timestep:1,encoder_hidden_states:1024x77x768"}});
      } else {
        OrtTensorRTProviderOptionsV2* trt_options;
        Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&trt_options));
        const std::vector<const char*> option_names{
            "trt_timing_cache_enable",
            "trt_dump_ep_context_model",
            "trt_ep_context_file_path",
            "trt_ep_context_embed_mode",
            "trt_weight_stripped_engine_enable",
        };
        std::string ctx_path = PathToUTF8(model_ctx);
        const std::vector<const char*> option_values{
            "1",                     // trt_timing_cache_enable
            "1",                     // trt_dump_ep_context_model
            ctx_path.c_str(),        // trt_ep_context_file_path
            "1",                     // trt_ep_context_embed_mode; 0 = context is engine cache path, 1 = context is engine binary data
            weightless ? "1" : "0",  // trt_weight_stripped_engine_enable
        };
        if (option_names.size() != option_values.size()) {
          throw std::runtime_error("wrong number of arguments");
        }
        Ort::ThrowOnError(api.UpdateTensorRTProviderOptions(trt_options, option_names.data(),
                                                            option_values.data(), option_names.size()));
        Ort::ThrowOnError(api.SessionOptionsAppendExecutionProvider_TensorRT_V2(so, trt_options));
      }
#if FILE_IO
      auto filebuf = readBinaryFile(model_file.string());
      std::vector<char> filebuf_data;
      if (!external_file_name.empty()) {
        filebuf_data = readBinaryFile(model_data_file.string());
      }
      std::vector<STRING_CLASS> file_names{external_file_name};
      std::vector<char*> file_buffers{filebuf_data.data()};
      std::vector<size_t> lengths{filebuf_data.size()};
      if (!external_file_name.empty()) {
        so.AddExternalInitializersFromFilesInMemory(file_names, file_buffers, lengths);
      }
      Ort::Session session_object(env, filebuf.data(), filebuf.size(), so);
#else
      Ort::Session session_object(env, model_file.c_str(), so);
#endif

      auto stop = std::chrono::high_resolution_clock::now();
      std::cout << "Session creation AOT: " << std::chrono::duration_cast<std::chrono::milliseconds>((stop - start)).count() << " ms" << std::endl;

      auto device_allocator = std::make_unique<Ort::Allocator>(session_object, mem_info_cuda);
      // OrtAllocator* allocator;
      // api.GetSharedAllocator(env, mem_info_cuda, &allocator);
      auto io_binding = generate_io_binding(session_object, *device_allocator, runtime_shape_override);
      Ort::RunOptions run_options;
      // run_options.AddConfigEntry(kOrtRunOptionsConfigDisableSynchronizeExecutionProviders, "1");
      run_options.AddConfigEntry(kOrtRunOptionsConfigEnableMemoryArenaShrinkage, "gpu:0");
      session_object.Run(run_options, io_binding);
    }

    // JIT time
    if (std::filesystem::exists(model_ctx)) {
      std::vector<char> filebuf, filebuf_data;
      auto start = std::chrono::high_resolution_clock::now();
      Ort::SessionOptions so;
      OrtTensorRTProviderOptionsV2* trt_options = nullptr;
      if (use_trt_rtx) {
        so.AppendExecutionProvider(onnxruntime::kNvTensorRTRTXExecutionProvider, {{onnxruntime::nv::provider_option_names::kHasUserComputeStream, "1"},
                                                                                  {onnxruntime::nv::provider_option_names::kUserComputeStream, stream_str},
                                                                                  {onnxruntime::nv::provider_option_names::kCudaGraphEnable, "0"}});
      } else {
        Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&trt_options));
      }
      std::vector<const char*> option_names;
      std::vector<const char*> option_values;
      if (!use_trt_rtx) {
        option_names.push_back("trt_weight_stripped_engine_enable");
        option_values.push_back("1");
        if (option_names.size() != option_values.size()) {
          throw std::runtime_error("wrong number of arguments");
        }
      }
#if FILE_IO
      char onnx_ptr_string[32];
      char extern_data_ptr_string[32];
      char onnx_size_string[32];
      char external_data_size_string[32];
      if (weightless) {
        filebuf = readBinaryFile(model_file.string());
        if (!external_file_name.empty()) {
          filebuf_data = readBinaryFile(model_data_file.string());
        }
#if WIN32
        sprintf_s(onnx_ptr_string, "%llu", reinterpret_cast<uint64_t>(filebuf.data()));
        sprintf_s(onnx_size_string, "%llu", static_cast<uint64_t>(filebuf.size()));
        sprintf_s(extern_data_ptr_string, "%llu", reinterpret_cast<uint64_t>(filebuf_data.data()));
        sprintf_s(external_data_size_string, "%llu", static_cast<uint64_t>(filebuf_data.size()));
#else
        sprintf(onnx_ptr_string, "%llu", reinterpret_cast<uint64_t>(filebuf.data()));
        sprintf(onnx_size_string, "%llu", static_cast<unsigned long>(filebuf.size()));
        sprintf(extern_data_ptr_string, "%llu", reinterpret_cast<uint64_t>(filebuf_data.data()));
        sprintf(external_data_size_string, "%llu", static_cast<uint64_t>(filebuf_data.size()));
#endif
        option_names.push_back("trt_onnx_bytestream");
        option_values.push_back(onnx_ptr_string);
        option_names.push_back("trt_onnx_bytestream_size");
        option_values.push_back(onnx_size_string);
        if (!external_file_name.empty()) {
          option_names.push_back("trt_external_data_bytestream");
          option_values.push_back(extern_data_ptr_string);
          option_names.push_back("trt_external_data_bytestream_size");
          option_values.push_back(external_data_size_string);
        }
      }
#else
      option_names.push_back("trt_onnx_model_folder_path");
      std::string path = model_dir.string();
      option_values.push_back(path.c_str());
#endif
      if (!use_trt_rtx) {
        Ort::ThrowOnError(api.UpdateTensorRTProviderOptions(trt_options, option_names.data(),
                                                            option_values.data(), option_names.size()));
        Ort::ThrowOnError(api.SessionOptionsAppendExecutionProvider_TensorRT_V2(so, trt_options));
      }
#if FILE_IO

      auto filebuf_ctx = readBinaryFile(model_ctx.string());
      Ort::Session session_object(env, filebuf_ctx.data(), filebuf_ctx.size(), so);
#else
      Ort::Session session_object(env, model_ctx.c_str(), so);
#endif

      auto stop = std::chrono::high_resolution_clock::now();
      std::cout << "Session creation JIT: " << std::chrono::duration_cast<std::chrono::milliseconds>((stop - start)).count() << " ms" << std::endl;
      auto device_allocator = std::make_unique<Ort::Allocator>(session_object, mem_info_cuda);

      auto io_binding = generate_io_binding(session_object, *device_allocator, runtime_shape_override);
      for (int i = 0; i < 100; ++i) {
        const int64_t batch_size = 2 + i * 2 % 32;
        std::unordered_map<std::string, std::vector<int64_t>> runtime_shape_override{
            {"sample", {batch_size, 4, 16, 16}},
            {"timestep", {1}},
            {"encoder_hidden_states", {batch_size, 77, 768}}};
        auto io_binding = generate_io_binding(session_object, *device_allocator, runtime_shape_override);
        Ort::RunOptions run_options;
        // run_options.AddConfigEntry(kOrtRunOptionsConfigDisableSynchronizeExecutionProviders, "1");
        // run_options.AddConfigEntry(kOrtRunOptionsConfigEnableMemoryArenaShrinkage, "gpu:0");
        session_object.Run(run_options, io_binding);
      }
    } else {
      throw std::runtime_error("Context file could not be generated.");
    }
  } catch (std::runtime_error& e) {
    std::cout << e.what() << std::endl;
  }
  return 0;
}
