// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/platform/ort_mutex.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cuda/cudnn_common.h"
#include "core/providers/cpu/nn/conv_attributes.h"
#include "core/providers/cuda/nn/conv.h"
#include <list>

namespace onnxruntime {
namespace contrib {
namespace cuda {

struct CudnnFrontEndConvState {
  // if x/w dims changed, update algo and cudnnTensors
  TensorShape last_x_dims;
  TensorShape last_w_dims;

  // these would be recomputed if x/w dims change
  TensorShape y_dims;
  TensorShapeVector y_dims_with_adjusted_pads;
  // size_t workspace_bytes;
  onnxruntime::cuda::CudnnTensor x_tensor;
  // const void* x_data = nullptr;
  size_t element_size = 0;
  onnxruntime::cuda::CudnnFilterDescriptor w_desc;
  // const void* w_data = nullptr;
  onnxruntime::cuda::CudnnTensor b_tensor;
  // const void* b_data = nullptr;
  void* b_zero = nullptr;
  onnxruntime::cuda::CudnnTensor y_tensor;
  Tensor* Y = nullptr;
  // const void* y_data = nullptr;
  // CudnnTensor z_tensor;
  // const void* z_data = nullptr;
  onnxruntime::cuda::CudnnConvolutionDescriptor conv_desc;

  // Some properties needed to support asymmetric padded Conv nodes
  bool post_slicing_required;
  TensorShapeVector slice_starts;
  TensorShapeVector slice_ends;
  TensorShapeVector slice_axes;

  // note that conv objects are shared between execution frames, and a lock is needed to avoid multi-thread racing
  OrtMutex mutex;
  IAllocatorUniquePtr<void> memory_for_cudnn_conv_results;

  ~CudnnFrontEndConvState() {
    if (b_zero) {
      CUDA_CALL_THROW(cudaFree(b_zero));
      b_zero = nullptr;
    }
  }
};


// ONNX Conv operator uses NCHW format for input, weights and output.
// NhwcConv contrib ops uses NHWC format: last dimension of input, weights and output are channels.
template <typename T, bool NHWC>
class NhwcFusedConvPW : public onnxruntime::cuda::CudaKernel {
 public:
  using CudaT = typename onnxruntime::cuda::ToCudaType<T>::MappedType;

  NhwcFusedConvPW(const OpKernelInfo& info) : CudaKernel(info), conv_attrs_(info) {
    auto pads_size = conv_attrs_.pads.size();
    ORT_ENFORCE(pads_size % 2 == 0);
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 protected:
  inline IAllocatorUniquePtr<void> GetWorkSpace(onnxruntime::Stream* stream) const {
    return GetScratchBuffer<void>(s_.workspace_bytes, stream);
  }

  Status UpdateState(OpKernelContext* context, bool bias_expected = false) const;
    constexpr static auto kDefaultConvAlgo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;

  ConvAttributes conv_attrs_;
  static const cudnnConvolutionFwdAlgo_t kAllAlgos[];
  mutable onnxruntime::cuda::CudnnConvState<cudnnConvolutionFwdAlgoPerf_t> s_;

  // cudnn frontend
  Status ComputeInternalCudnnFrontEnd(OpKernelContext* context) const;
  Status FinalizeGraph(OpKernelContext* context, std::vector<cudnn_frontend::Operation const*> & ops) const;

  mutable std::vector<cudnn_frontend::ManagedOpaqueDescriptor> plan_desc;
  mutable CudnnFrontEndConvState sf_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
