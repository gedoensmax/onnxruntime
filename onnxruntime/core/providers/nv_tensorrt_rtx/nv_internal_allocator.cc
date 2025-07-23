// Copyright (c) Microsoft Corporation. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Licensed under the MIT License.

#include "nv_internal_allocator.h"

#include "core/providers/cuda/shared_inc/cuda_call.h"
#include "core/providers/shared_library/provider_api.h"

#define LOG_ALLOCATIONS 0

namespace onnxruntime {
TRTAllocatorAsync::TRTAllocatorAsync(std::string name, OrtAllocator* base_allocator)
    : name_(name),
      allocator_(base_allocator) {
}

void* TRTAllocatorAsync::allocateAsync(uint64_t const size, [[maybe_unused]] uint64_t const alignment, [[maybe_unused]] nvinfer1::AllocatorFlags const flags,
                                       cudaStream_t stream) noexcept {
  void* new_allocation = nullptr;
  if (allocator_) {
#if LOG_ALLOCATIONS
    LOGS_DEFAULT(VERBOSE) << "[NvTensorRTRTX EP] Allocator with name " << name_ << " allocated " << size << " through OrtAllocator";
#endif
    new_allocation = allocator_->Alloc(allocator_, size);
  } else {
    if (cudaMallocAsync(&new_allocation, size, stream) != cudaSuccess) {
      LOGS_DEFAULT(ERROR) << "[NvTensorRTRTX EP] Allocator with name " << name_ << " failed to allocate " << size;
      return nullptr;
    }
#if LOG_ALLOCATIONS
    LOGS_DEFAULT(VERBOSE) << "[NvTensorRTRTX EP] Allocator with name " << name_ << " allocated " << size;
#endif
  }
  {
    auto lock = std::scoped_lock(map_mutex);
    buffer_map_.insert({new_allocation, size});
  }
  return new_allocation;
}
void* TRTAllocatorAsync::reallocate([[maybe_unused]] void* const baseAddr, [[maybe_unused]] uint64_t alignment, [[maybe_unused]] uint64_t newSize) noexcept {
  auto lock = std::scoped_lock(map_mutex);
  // TODO: returning a nullptr will tell TRT that this is not implemented
  return nullptr;
}

bool TRTAllocatorAsync::deallocateAsync(void* const memory, cudaStream_t stream) noexcept {
  {
    auto lock = std::scoped_lock(map_mutex);
    auto it = buffer_map_.find(memory);
    if (it == buffer_map_.end()) {
      return false;
    }
    buffer_map_.erase(it);
  }
  if (allocator_) {
    allocator_->Free(allocator_, memory);
#if LOG_ALLOCATIONS
    LOGS_DEFAULT(VERBOSE) << "[NvTensorRTRTX EP] Allocator with name " << name_ << " freed allocation through OrtAllocator";
#endif
  } else {
    CUDA_CALL_THROW(cudaFreeAsync(memory, stream));
#if LOG_ALLOCATIONS
    LOGS_DEFAULT(VERBOSE) << "[NvTensorRTRTX EP] Allocator with name " << name_ << " freed allocation";
#endif
  }
  return true;
}

void TRTAllocatorAsync::logStats() {
  LOGS_DEFAULT(VERBOSE) << "[NvTensorRTRTX EP] Allocator with name " << name_ << " has allocations:\n";
  for (auto& [ptr, size] : buffer_map_) {
    LOGS_DEFAULT(VERBOSE) << "\t" << "adress:" << ptr << " holds " << size << " bytes\n";
  }
}

}  // namespace onnxruntime