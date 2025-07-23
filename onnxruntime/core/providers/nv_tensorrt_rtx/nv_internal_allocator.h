// Copyright (c) Microsoft Corporation. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "nv_includes.h"
#include "core/framework/allocator.h"

#include <mutex>

namespace onnxruntime {

class TRTAllocatorAsync : public nvinfer1::IGpuAsyncAllocator {
 public:
  TRTAllocatorAsync() = delete;
  TRTAllocatorAsync(std::string name, OrtAllocator* base_allocator = nullptr);

  void* allocateAsync(uint64_t const size, uint64_t const alignment, nvinfer1::AllocatorFlags const flags,
                      cudaStream_t /*stream*/) noexcept override;

  void* reallocate(void* const, uint64_t, uint64_t) noexcept override;

  bool deallocateAsync(void* const memory, cudaStream_t /*stream*/) noexcept override;

  void logStats();

 private:
  std::string name_;
  OrtAllocator* const allocator_;
  std::mutex map_mutex;
  std::unordered_map<void*, size_t> buffer_map_;
};
}  // namespace onnxruntime
