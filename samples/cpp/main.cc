// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_run_options_config_keys.h>
#include <onnxruntime_session_options_config_keys.h>
#include <core/graph/constants.h>
#include <vector>
#include <filesystem>
#include <iostream>
#include <chrono>
#include <fstream>
#include <codecvt>

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

Ort::IoBinding generate_io_binding(Ort::Session& session) {
  Ort::IoBinding binding(session);
  auto allocator = Ort::AllocatorWithDefaultOptions();
  for (int input_idx = 0; input_idx < int(session.GetInputCount()); ++input_idx) {
    auto input_name = session.GetInputNameAllocated(input_idx, Ort::AllocatorWithDefaultOptions());
    auto full_tensor_info = session.GetInputTypeInfo(input_idx);
    auto tensor_info = full_tensor_info.GetTensorTypeAndShapeInfo();
    auto shape = tensor_info.GetShape();
    auto type = tensor_info.GetElementType();
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
    binding.BindOutput(output_name.get(), allocator.GetInfo());
  }
  return binding;
}

int main() {
  auto logging_level = OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING;
  //    auto logging_level =OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE;
  try {
    std::filesystem::path model_file = "test_onnx/sd.onnx";
    std::string external_file_name = "sd.onnx_data";
    std::filesystem::path model_data = "test_onnx";
    model_data /= external_file_name;
    // std::filesystem::path model_name = "test_onnx/resnet101-v2-7_bs8.onnx";
    if (!std::filesystem::exists(model_file)) {
      throw std::runtime_error("file does not exist");
    }
    std::filesystem::path model_ctx = "/tmp/model_ctx.onnx";
    if (std::filesystem::exists(model_ctx)) {
      std::filesystem::remove(model_ctx);
    }
    auto env = Ort::Env();
    auto api = Ort::GetApi();
    env.UpdateEnvWithCustomLogLevel(logging_level);

    // AOT time
    {
      auto start = std::chrono::high_resolution_clock::now();
      Ort::SessionOptions so;
      Ort::RunOptions run_options;
      // {sample [dtype=float32, shape=('2B', 4, 'H', 'W')],
      //  timestep [dtype=float32, shape=(1,)],
      //  encoder_hidden_states [dtype=float16, shape=('2B', 77, 768)]}
      Ort::ThrowOnError(api.AddFreeDimensionOverrideByName(so, "2B", 2));
      Ort::ThrowOnError(api.AddFreeDimensionOverrideByName(so, "H", 64));
      Ort::ThrowOnError(api.AddFreeDimensionOverrideByName(so, "W", 64));

      so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
      so.AddConfigEntry(kOrtSessionOptionEpContextEmbedMode, "1");
      so.AddConfigEntry("ep.context_file_path", PathToUTF8(model_ctx).c_str());
      so.AppendExecutionProvider(onnxruntime::kNvTensorRTRTXExecutionProvider, {});

      Ort::Session session_object(env, model_file.c_str(), so);
      /*
            auto filebuf = readBinaryFile(model_file.string());
            auto filebuf_data = readBinaryFile(model_data.string());
            std::vector<std::string> file_names{external_file_name};
            std::vector<char*> file_buffers{filebuf_data.data()};
            std::vector<size_t> lengths{filebuf_data.size()};
            so.AddExternalInitializersFromFilesInMemory(file_names, file_buffers, lengths);
            Ort::Session session_object(env, filebuf.data(), filebuf.size(), so);
      */
      auto stop = std::chrono::high_resolution_clock::now();
      std::cout << "Session creation AOT: " << std::chrono::duration_cast<std::chrono::milliseconds>((stop - start)).count() << " ms" << std::endl;

      auto io_binding = generate_io_binding(session_object);
      session_object.Run(run_options, io_binding);
    }

    // JIT time
    {
      auto start = std::chrono::high_resolution_clock::now();
      Ort::SessionOptions so;
      Ort::RunOptions run_options;
      so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
      so.AppendExecutionProvider(onnxruntime::kNvTensorRTRTXExecutionProvider, {});
      Ort::Session session_object(env, model_ctx.c_str(), so);
      auto stop = std::chrono::high_resolution_clock::now();
      std::cout << "Session creation JIT: " << std::chrono::duration_cast<std::chrono::milliseconds>((stop - start)).count() << " ms" << std::endl;

      auto io_binding = generate_io_binding(session_object);
      session_object.Run(run_options, io_binding);
    }
  } catch (std::runtime_error& e) {
    std::cout << e.what() << std::endl;
  }
  return 0;
}
