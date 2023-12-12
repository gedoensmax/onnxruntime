// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.


#include "contrib_ops/cuda/conv_pw.h"
#include "core/common/span_utils.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_pch.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "core/providers/cuda/tensor/slice.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

static std::vector<int64_t> generateStrides(const onnxruntime::TensorShapeVector& shape, bool channels_last) {
  // For INT8x4 and INT8x32 we still compute standard strides here to input
  // into the cuDNN functions. We will manually scale by resizeFactor in the cpu ref.
  std::vector<int64_t> strides;
  strides.resize(shape.size());
  int nbDims = strides.size();
  if (channels_last) {
    // Here we assume that the format is CUDNN_TENSOR_NHWC
    strides[1] = 1;
    strides[nbDims - 1] = strides[1] * shape[1];
    for (int64_t d = nbDims - 2; d >= 2; d--) {
      strides[d] = strides[d + 1] * shape[d + 1];
    }
    strides[0] = strides[2] * shape[2];
  } else {
    strides[nbDims - 1] = 1;
    for (int64_t d = nbDims - 2; d >= 0; d--) {
      strides[d] = strides[d + 1] * shape[d + 1];
    }
  }
  return strides;
}

template <typename T, bool NHWC>
const cudnnConvolutionFwdAlgo_t NhwcFusedConvPW<T, NHWC>::kAllAlgos[] = {
    CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
    CUDNN_CONVOLUTION_FWD_ALGO_FFT,
    CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
    CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
};

static cudnnStatus_t GetWorkspaceSize(cudnnHandle_t handle, const CudnnConvState<cudnnConvolutionFwdAlgoPerf_t>& s, cudnnConvolutionFwdAlgo_t algo, size_t* sz) {
  return cudnnGetConvolutionForwardWorkspaceSize(handle, s.x_tensor, s.w_desc, s.conv_desc, s.y_tensor, algo, sz);
}

static size_t GetMaxWorkspaceSize(cudnnHandle_t handle, const CudnnConvState<cudnnConvolutionFwdAlgoPerf_t>& s,
                           const cudnnConvolutionFwdAlgo_t* algo, int n_algo) {
  // TODO: get maximum available size from memory arena
  size_t free, total;
  CUDA_CALL_THROW(cudaMemGetInfo(&free, &total));
  // Assuming 10% of fragmentation
  free = static_cast<size_t>(static_cast<double>(free) * 0.9);
  size_t max_ws_size = 0;
  for (int i = 0; i < n_algo; i++) {
    cudnnStatus_t err;
    size_t sz;
    err = GetWorkspaceSize(handle, s, algo[i], &sz);
    if (CUDNN_STATUS_SUCCESS != err || sz == 0 || sz < max_ws_size || sz > free) continue;
    max_ws_size = sz;
  }
  return max_ws_size;
}

template <typename T, bool NHWC>
Status NhwcFusedConvPW<T, NHWC>::ComputeInternalCudnnFrontEnd(OpKernelContext* context) const {
  std::lock_guard<OrtMutex> lock(s_.mutex);
  ORT_RETURN_IF_ERROR(UpdateState(context));
  if (s_.Y->Shape().Size() == 0) {
    return Status::OK();
  }
  IAllocatorUniquePtr<void> workspace = GetWorkSpace(context->GetComputeStream());
  const void* data_ptrs[3] = {s_.x_data, s_.y_data,
                              s_.w_data};
  int64_t uids[] = {'x', 'y', 'w'};
  auto variant_pack = cudnn_frontend::VariantPackBuilder()
                          .setWorkspacePointer(workspace.get())
                          .setDataPointers(3, const_cast<void**>(data_ptrs))
                          .setUids(3, uids)
                          .build();
  if (s_.b_data) {
    const void* data_ptrs[4] = {s_.x_data, s_.y_data,
                                s_.w_data, s_.b_data};
    int64_t uids[] = {'x', 'y', 'w', 'b'};
    variant_pack = cudnn_frontend::VariantPackBuilder()
                       .setWorkspacePointer(workspace.get())
                       .setDataPointers(4, const_cast<void**>(data_ptrs))
                       .setUids(4, uids)
                       .build();
  }
  auto cudnn_handle = GetCudnnHandle(context);
  // TODO(maximilianm) here we possibly have to iterate over found plans and variant packs
  for (auto& plan : plan_desc) {
    auto status = cudnnBackendExecute(cudnn_handle, plan->get_backend_descriptor(), variant_pack.get_raw_desc());
    if (status != CUDNN_STATUS_SUCCESS) {
      ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Plan execute error");
    }
  }
  // To deal with asymmetric padding, we may have over-padded on one or both sides of the spatial dimensions
  // This may have lead to extra results that are unnecessary and hence we slice that off here
  if (sf_.post_slicing_required) {
    ORT_RETURN_IF_ERROR(SliceOutUnwantedOutputSection(Stream(context), s_.y_data, gsl::make_span(sf_.y_dims_with_adjusted_pads),
                                                      sf_.Y->MutableDataRaw(), sf_.y_dims.GetDims(), sf_.slice_starts,
                                                      sf_.slice_ends, sf_.slice_axes, sf_.element_size));
  }
  return Status::OK();
}

template <typename T, bool NHWC>
Status NhwcFusedConvPW<T, NHWC>::FinalizeGraph(OpKernelContext* context, std::vector<cudnn_frontend::Operation const*>& ops) const {
  auto cudnn_handle = GetCudnnHandle(context);
  auto opGraph = cudnn_frontend::OperationGraphBuilder()
                     .setHandle(cudnn_handle)
                     .setOperationGraph(ops.size(), ops.data())
                     .build();
  // search configs
  cudnn_frontend::EngineConfigList filtered_configs;
  auto statuses =
      cudnn_frontend::get_heuristics_list<2>(
          {"heuristics_instant",
           "heuristics_fallback"},
          opGraph, [](cudnnBackendDescriptor_t engine_config) { return false; }, filtered_configs);

  // TODO if no configs are found already fallback here

  // TODO(maximilianm) what can we reuse if only spatial dimensions of the input change but we can reuse the actual kernel?
  // build plan
  for (auto& config : filtered_configs) {
    try {
      auto plan =
          cudnn_frontend::ExecutionPlanBuilder().setHandle(cudnn_handle).setEngineConfig(config, opGraph.getTag()).build();
      // std::cout << "Plan tag: " << plan.getTag() << std::endl;

      s_.workspace_bytes = plan.getWorkspaceSize();
      // std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;
      plan_desc = {plan.get_desc()};
    } catch (cudnn_frontend::cudnnException& e) {
      // status = e.getCudnnStatus();
      continue;
    }
  }
  if (plan_desc.empty()) {
    s_.workspace_bytes = 0;
    try {
      // TODO this fallback approach does not work as we still have virtual tensors ...
      for (int op_idx = 0; op_idx < ops.size(); ++op_idx) {
        auto statuses = cudnn_frontend::get_heuristics_list<2>(
            {"heuristics_instant",
             "heuristics_fallback"},
            opGraph, [](cudnnBackendDescriptor_t engine_config) { return false; }, filtered_configs);
        std::cout << "Compiling fallback for node: " << ops[op_idx]->describe() << std::endl;
        opGraph = cudnn_frontend::OperationGraphBuilder()
                      .setHandle(cudnn_handle)
                      .setOperationGraph(1, &ops[op_idx])
                      .build();
        auto plan =
            cudnn_frontend::ExecutionPlanBuilder().setHandle(cudnn_handle).setEngineConfig(filtered_configs[0], opGraph.getTag()).build();
        s_.workspace_bytes = std::max(static_cast<size_t>(plan.getWorkspaceSize()), s_.workspace_bytes);
        plan_desc.push_back(plan.get_desc());
      }
    } catch (cudnn_frontend::cudnnException& e) {
      std::cout << "[ERROR] Exception " << e.what() << " " << cudnn_frontend::to_string(e.getCudnnStatus()) << std::endl;
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "No plan found implementing the operation graph");
    }
  }
  return Status::OK();
}

template <typename T, bool NHWC>
Status NhwcFusedConvPW<T, NHWC>::UpdateState(OpKernelContext* context, bool bias_expected) const {
  // set X
  const Tensor* X = context->Input<Tensor>(0);
  const TensorShape& x_shape = X->Shape();
  const auto x_dims = x_shape.AsShapeVector();
  s_.x_data = reinterpret_cast<const CudaT*>(X->Data<T>());
  s_.element_size = X->DataType()->Size();
  // set W
  const Tensor* W = context->Input<Tensor>(1);
  const TensorShape& w_shape = W->Shape();
  auto w_dims = w_shape.AsShapeVector();
  s_.w_data = reinterpret_cast<const CudaT*>(W->Data<T>());

  // Make sure input and weight are 4D for NHWC since we set 4D descriptor for NHWC.
  constexpr bool channels_last = NHWC;
  if (channels_last && (x_shape.NumDimensions() != 4 || w_shape.NumDimensions() != 4)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Number of dimensions of X and W should be 4 for channels_last format (NHWC)");
  }

  // set B
  if (context->InputCount() >= 3) {
    const Tensor* B = context->Input<Tensor>(2);
    s_.b_data = reinterpret_cast<const CudaT*>(B->Data<T>());
  } else {
    s_.b_data = nullptr;
  }
  // set Z
  if (context->InputCount() >= 4) {
    const Tensor* Z = context->Input<Tensor>(3);
    ORT_RETURN_IF_ERROR(s_.z_tensor.Set(Z->Shape().GetDims(), CudnnTensor::GetDataType<CudaT>()));
    s_.z_data = reinterpret_cast<const CudaT*>(Z->Data<T>());
  } else {
    s_.z_data = nullptr;
  }
  bool input_dims_changed = (s_.last_x_dims != x_dims);
  bool w_dims_changed = (s_.last_w_dims != w_dims);
  if (input_dims_changed || w_dims_changed) {
    if (input_dims_changed)
      s_.last_x_dims = gsl::make_span(x_dims);

    if (w_dims_changed) {
      s_.last_w_dims = gsl::make_span(w_dims);
      s_.cached_benchmark_results.clear();
    }

    ORT_RETURN_IF_ERROR(conv_attrs_.ValidateInputShape(X->Shape(), W->Shape(), channels_last, channels_last));

    TensorShapeVector kernel_shape;
    ORT_RETURN_IF_ERROR(conv_attrs_.ComputeKernelShape(W->Shape(), kernel_shape, channels_last));

    const size_t kernel_rank = kernel_shape.size();

    ConvPadVector pads(conv_attrs_.pads);
    if (pads.empty()) {
      pads.resize(kernel_rank * 2, 0);
    }
    TensorShapeVector dilations(conv_attrs_.dilations);
    if (dilations.empty()) {
      dilations.resize(kernel_rank, 1);
    }
    TensorShapeVector strides(conv_attrs_.strides);
    if (strides.empty()) {
      strides.resize(kernel_rank, 1);
    }

    TensorShapeVector y_dims;
    y_dims.reserve(2 + kernel_rank);  // add 2 to account for 'N' and 'C'

    const int64_t N = X->Shape()[0];
    const int64_t M = W->Shape()[0];
    if (channels_last) {
      y_dims.push_back(N);
    } else {
      y_dims.insert(y_dims.begin(), {N, M});
    }

    bool post_slicing_required = false;
    TensorShapeVector slice_starts;
    slice_starts.reserve(kernel_rank);

    TensorShapeVector slice_ends;
    slice_ends.reserve(kernel_rank);

    TensorShapeVector slice_axes;
    slice_axes.reserve(kernel_rank);

    constexpr size_t spatial_dim_start = channels_last ? 1 : 2;
    const size_t spatial_dim_end = spatial_dim_start + kernel_rank;
    TensorShape spatial_shape = X->Shape().Slice(spatial_dim_start, spatial_dim_end);

    TensorShapeVector y_dims_with_adjusted_pads(y_dims);
    ORT_RETURN_IF_ERROR(conv_attrs_.InferOutputShapeWithAdjustedPads(spatial_shape, kernel_shape,
                                                                     strides, dilations, pads, y_dims, y_dims_with_adjusted_pads,
                                                                     post_slicing_required, slice_starts, slice_ends, slice_axes,
                                                                     channels_last));
    if (channels_last) {
      y_dims.push_back(M);
      y_dims_with_adjusted_pads.push_back(M);
    }

    ORT_ENFORCE(y_dims.size() == y_dims_with_adjusted_pads.size());
    s_.y_dims = gsl::make_span(y_dims);
    s_.y_dims_with_adjusted_pads = y_dims_with_adjusted_pads;
    s_.post_slicing_required = post_slicing_required;
    s_.slice_starts = slice_starts;
    s_.slice_ends = slice_ends;
    s_.slice_axes = slice_axes;

    s_.Y = context->Output(0, TensorShape(s_.y_dims));
    if (post_slicing_required) {
      // Post slicing needed. Create and fill in the Conv results in an intermediate buffer.
      s_.memory_for_cudnn_conv_results = GetScratchBuffer<void>(TensorShape(y_dims_with_adjusted_pads).Size() * s_.element_size, context->GetComputeStream());
      s_.y_data = reinterpret_cast<CudaT*>(s_.memory_for_cudnn_conv_results.get());
    } else {
      // No post slicing needed. Fill the output tensor's buffer directly.
      s_.y_data = reinterpret_cast<CudaT*>(s_.Y->MutableData<T>());
    }

    const CUDAExecutionProvider* cuda_ep =
        static_cast<const CUDAExecutionProvider*>(this->Info().GetExecutionProvider());

    TensorShapeVector x_dims_cudnn{x_dims.begin(), x_dims.end()};
    TensorShapeVector y_dims_cudnn = !post_slicing_required ? y_dims : y_dims_with_adjusted_pads;
    if (kernel_rank < 2) {
      // TODO: Explore padding the provided input shape [N, C, D] to [N, C, 1, D]
      // especially for EXHAUSTIVE algo search which may result in a better algo selection.
      // ORTModule uses different algo search options (HEURISTIC, and use max workspace size) compared to
      // inference build (EXHAUSTIVE, 32M workspace size). We observed better perf when we pad input shape
      // [N,C,D] to [N,C,1,D], expecially on A100, and especially for ConvGrad.
      // PyTorch also pads to [N,C,1,D]. For inference build, we still pad it to [N, C, D, 1] as this seems
      // to be the sweet spot for all algo search options: EXHAUSTIVE, HEURISTIC, and DEFAULT.
      // See PR #7348 and #7702 for more context.
      if (cuda_ep->GetCudnnConv1dPadToNc1d()) {
        x_dims_cudnn.insert(x_dims_cudnn.begin() + 2, 1);
        y_dims_cudnn.insert(y_dims_cudnn.begin() + 2, 1);
        w_dims.insert(w_dims.begin() + 2, 1);
        pads.insert(pads.begin() + kernel_rank, 0);
        pads.insert(pads.begin(), 0);
        kernel_shape.insert(kernel_shape.begin(), 1);
        strides.insert(strides.begin(), 1);
        dilations.insert(dilations.begin(), 1);
      } else {
        x_dims_cudnn.push_back(1);
        y_dims_cudnn.push_back(1);
        w_dims.push_back(1);
        pads.insert(pads.begin() + kernel_rank, 0);
        pads.insert(pads.end(), 0);
        kernel_shape.push_back(1);
        strides.push_back(1);
        dilations.push_back(1);
      }
    }

    int cudnn_conv_algo_ = cuda_ep->GetCudnnConvAlgo();
    std::cout << "cudnn_conv_algo_" << cudnn_conv_algo_ << std::endl;
    if (cudnn_conv_algo_ == OrtCudnnConvAlgoGraph) {
      constexpr auto compute_dtype = CUDNN_DATA_FLOAT;
      const auto store_dtype = CudnnTensor::GetDataType<CudaT>();

      auto x_strides = generateStrides(x_dims_cudnn, channels_last);
      auto x_tensor = cudnn_frontend::TensorBuilder()
                          .setDim(x_dims_cudnn.size(), x_dims_cudnn.data())
                          .setStride(x_dims_cudnn.size(), x_strides.data())
                          .setId('x')
                          .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                          .setDataType(store_dtype)
                          .build();

      // TODO check y_dims calculation
      auto y_strides = generateStrides(y_dims_cudnn, channels_last);
      auto y_tensor = cudnn_frontend::TensorBuilder()
                          .setDim(y_dims_cudnn.size(), y_dims_cudnn.data())
                          .setStride(y_dims_cudnn.size(), y_strides.data())
                          .setId('y')
                          .setAlignment(16)
                          .setDataType(store_dtype)
                          .build();

      auto w_strides = generateStrides(w_dims, channels_last);
      auto w_tensor = cudnn_frontend::TensorBuilder()
                          .setDim(w_dims.size(), w_dims.data())
                          .setStride(w_dims.size(), w_strides.data())
                          .setId('w')
                          .setAlignment(16)
                          .setDataType(store_dtype)
                          .build();
      // Define the convolution problem
      constexpr int conv_dim = 2;
      const int64_t padding[2] = {0, 0};
      auto conv_desc = cudnn_frontend::ConvDescBuilder()
                           .setComputeType(compute_dtype)
                           .setMathMode(CUDNN_CROSS_CORRELATION)
                           .setSpatialDimCount(conv_dim)
                           .setSpatialStride(conv_dim, strides.data())
                           .setPrePadding(conv_dim, pads.data())  // TODO check padding
                           .setPostPadding(conv_dim, padding)
                           .setDilation(conv_dim, dilations.data())
                           .build();
      std::cout << x_tensor.describe() << std::endl;
      std::cout << w_tensor.describe() << std::endl;
      if (context->InputCount() >= 3) {
      }
      std::cout << y_tensor.describe() << std::endl;
      std::cout << conv_desc.describe() << std::endl;

      float alpha = 1.0f;
      float beta = 0.0f;
      std::vector<cudnn_frontend::Operation const*> ops;
      s_.b_data = nullptr;
      if (context->InputCount() >= 3) {
        // set B
        const Tensor* B = context->Input<Tensor>(2);
        const TensorShape& b_shape = B->Shape();
        TensorShapeVector b_dims(y_dims.size(), 1);
        b_dims[1] = b_shape[0];
        s_.b_data = reinterpret_cast<const CudaT*>(B->Data<T>());
        auto b_strides = generateStrides(b_dims, channels_last);
        auto b_tensor = cudnn_frontend::TensorBuilder()
                            .setDim(b_dims.size(), b_dims.data())
                            .setStride(b_strides.size(), b_strides.data())
                            .setId('b')
                            .setAlignment(16)
                            .setDataType(store_dtype)
                            .build();
        std::cout << b_tensor.describe() << std::endl;

        auto conv_out_tensor = cudnn_frontend::TensorBuilder()
                                   .setDim(y_dims_cudnn.size(), y_dims_cudnn.data())
                                   .setStride(y_dims_cudnn.size(), y_strides.data())
                                   .setId('t')
                                   .setAlignment(16)
                                   .setDataType(store_dtype)
                                   .setVirtual()
                                   .build();

        auto biasDesc = cudnn_frontend::PointWiseDescBuilder()
                            .setMode(CUDNN_POINTWISE_ADD)
                            .setComputeType(compute_dtype)
                            .build();

        // Create a convolution Node
        auto conv_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                           .setxDesc(x_tensor)
                           .setwDesc(w_tensor)
                           .setyDesc(conv_out_tensor)
                           .setcDesc(conv_desc)
                           .setAlpha(alpha)
                           .setBeta(beta)
                           .build();
        std::cout << conv_op.describe() << std::endl;

        auto bias_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                           .setxDesc(conv_out_tensor)
                           .setbDesc(b_tensor)
                           .setyDesc(y_tensor)
                           .setpwDesc(biasDesc)
                           .build();
        std::cout << bias_op.describe() << std::endl;

        ops = {&conv_op, &bias_op};
        return FinalizeGraph(context, ops);
      } else {
        // Create a convolution Node
        auto conv_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                           .setxDesc(x_tensor)
                           .setwDesc(w_tensor)
                           .setyDesc(y_tensor)
                           .setcDesc(conv_desc)
                           .setAlpha(alpha)
                           .setBeta(beta)
                           .build();
        ops = {&conv_op};
        return FinalizeGraph(context, ops);
      }
    }

    if (w_dims_changed) {
      if (!channels_last) {
        ORT_RETURN_IF_ERROR(s_.w_desc.Set(w_dims, CudnnTensor::GetDataType<CudaT>()));
      } else {
        ORT_RETURN_IF_ERROR(s_.w_desc.Set(CUDNN_TENSOR_NHWC,
                                          CudnnTensor::GetDataType<CudaT>(),
                                          static_cast<int>(w_dims[0]),
                                          static_cast<int>(w_dims[3]),
                                          static_cast<int>(w_dims[1]),
                                          static_cast<int>(w_dims[2])));
      }
    }

    // We must delay returning early until here so that the weight dims have been cached properly
    if (s_.Y->Shape().Size() == 0) {
      return Status::OK();
    }

    if (channels_last) {
      ORT_RETURN_IF_ERROR(s_.x_tensor.Set(CUDNN_TENSOR_NHWC,
                                          CudnnTensor::GetDataType<CudaT>(),
                                          static_cast<int>(x_dims_cudnn[0]),
                                          static_cast<int>(x_dims_cudnn[3]),
                                          static_cast<int>(x_dims_cudnn[1]),
                                          static_cast<int>(x_dims_cudnn[2])));

      ORT_RETURN_IF_ERROR(s_.y_tensor.Set(CUDNN_TENSOR_NHWC,
                                          CudnnTensor::GetDataType<CudaT>(),
                                          static_cast<int>(y_dims_cudnn[0]),
                                          static_cast<int>(y_dims_cudnn[3]),
                                          static_cast<int>(y_dims_cudnn[1]),
                                          static_cast<int>(y_dims_cudnn[2])));
    } else {
      ORT_RETURN_IF_ERROR(s_.x_tensor.Set(x_dims_cudnn, CudnnTensor::GetDataType<CudaT>()));
      ORT_RETURN_IF_ERROR(s_.y_tensor.Set(y_dims_cudnn, CudnnTensor::GetDataType<CudaT>()));
    }

    ORT_RETURN_IF_ERROR(s_.conv_desc.Set(kernel_shape.size(), pads, strides, dilations,
                                         gsl::narrow_cast<int>(conv_attrs_.group),
                                         CUDNN_CROSS_CORRELATION, CudnnTensor::GetDataType<CudaT>()));

    if (context->InputCount() >= 3) {
      const Tensor* B = context->Input<Tensor>(2);
      const auto& b_shape = B->Shape();
      ORT_RETURN_IF_NOT(b_shape.NumDimensions() == 1, "bias should be 1D");
      TensorShapeVector b_dims(2 + kernel_shape.size(), 1);
      b_dims[1] = b_shape[0];
      ORT_RETURN_IF_ERROR(s_.b_tensor.Set(b_dims, CudnnTensor::GetDataType<CudaT>()));
      // s_.b_data = reinterpret_cast<const CudaT*>(B->Data<T>());
    } else if (bias_expected) {
      TensorShapeVector b_dims(2 + kernel_shape.size(), 1);
      b_dims[1] = w_dims[0];
      auto malloc_size = b_dims[1] * sizeof(CudaT);
      ORT_RETURN_IF_ERROR(s_.b_tensor.Set(b_dims, CudnnTensor::GetDataType<CudaT>()));
      if (s_.b_zero) {
        CUDA_CALL_THROW(cudaFree(s_.b_zero));
        s_.b_zero = nullptr;
      }
      CUDA_CALL_THROW(cudaMalloc(&s_.b_zero, malloc_size));
      CUDA_CALL_THROW(cudaMemsetAsync(s_.b_zero, 0, malloc_size, Stream(context)));
    }

    if (!s_.cached_benchmark_results.contains(x_dims_cudnn)) {
      // set math type to tensor core before algorithm search
      if constexpr (std::is_same<T, MLFloat16>::value)
        CUDNN_RETURN_IF_ERROR(cudnnSetConvolutionMathType(s_.conv_desc, CUDNN_TENSOR_OP_MATH));

      cudnnConvolutionFwdAlgoPerf_t perf;
      int algo_count = 1;
      int cudnn_conv_algo = cuda_ep->GetCudnnConvAlgo();
      ORT_ENFORCE(cudnn_conv_algo > -1 && cudnn_conv_algo < 3, "cudnn_conv_algo should be 0, 1 or 2, but got ", cudnn_conv_algo);
      switch (cudnn_conv_algo) {
        case OrtCudnnConvAlgoSearchExhaustive: {
          static constexpr int num_algos = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
          size_t max_ws_size = cuda_ep->GetCudnnConvUseMaxWorkspace() ? GetMaxWorkspaceSize(GetCudnnHandle(context), s_, kAllAlgos, num_algos)
                                                                      : AlgoSearchWorkspaceSize;
          // Use GetTransientScratchBuffer() so the workspace can be freed instead of cached.
          // Because the benchmarking uses a huge amount of memory, e.g. a few GBs.
          IAllocatorUniquePtr<void> algo_search_workspace = GetTransientScratchBuffer<void>(max_ws_size);
          CUDNN_RETURN_IF_ERROR(cudnnFindConvolutionForwardAlgorithmEx(
              GetCudnnHandle(context),
              s_.x_tensor,
              s_.x_data,
              s_.w_desc,
              s_.w_data,
              s_.conv_desc,
              s_.y_tensor,
              s_.y_data,
              1,            // requestedAlgoCount
              &algo_count,  // returnedAlgoCount
              &perf,
              algo_search_workspace.get(),
              max_ws_size));
          break;
        }
        case OrtCudnnConvAlgoSearchHeuristic:
          CUDNN_RETURN_IF_ERROR(cudnnGetConvolutionForwardAlgorithm_v7(
              GetCudnnHandle(context),
              s_.x_tensor,
              s_.w_desc,
              s_.conv_desc,
              s_.y_tensor,
              1,            // requestedAlgoCount
              &algo_count,  // returnedAlgoCount
              &perf));
          break;

        default:
          perf.algo = kDefaultConvAlgo;
          CUDNN_RETURN_IF_ERROR(GetWorkspaceSize(GetCudnnHandle(context), s_, perf.algo, &perf.memory));
          if (std::is_same<T, MLFloat16>::value) {
            perf.mathType = CUDNN_TENSOR_OP_MATH;
          } else {
            perf.mathType = CUDNN_DEFAULT_MATH;
          }
      }
      s_.cached_benchmark_results.insert(x_dims_cudnn, {perf.algo, perf.memory, perf.mathType});
    }
    const auto& perf = s_.cached_benchmark_results.at(x_dims_cudnn);
    CUDNN_RETURN_IF_ERROR(cudnnSetConvolutionMathType(s_.conv_desc, perf.mathType));
    s_.algo = perf.algo;
    s_.workspace_bytes = perf.memory;
  } else {
    // set Y
    s_.Y = context->Output(0, s_.y_dims);
    if (s_.Y->Shape().Size() == 0) {
      return Status::OK();
    }
    if (s_.post_slicing_required) {
      s_.memory_for_cudnn_conv_results = GetScratchBuffer<void>(TensorShape(s_.y_dims_with_adjusted_pads).Size() * s_.element_size, context->GetComputeStream());
      s_.y_data = reinterpret_cast<CudaT*>(s_.memory_for_cudnn_conv_results.get());
    } else {
      s_.y_data = reinterpret_cast<CudaT*>(s_.Y->MutableData<T>());
    }
  }
  return Status::OK();
}

template <typename T, bool NHWC>
Status NhwcFusedConvPW<T, NHWC>::ComputeInternal(OpKernelContext* context) const {
  const CUDAExecutionProvider* cuda_ep_ =
      static_cast<const CUDAExecutionProvider*>(this->Info().GetExecutionProvider());
  int cudnn_conv_algo_ = cuda_ep_->GetCudnnConvAlgo();
  if (cudnn_conv_algo_ == OrtCudnnConvAlgoGraph) {
    return ComputeInternalCudnnFrontEnd(context);
  }
  std::lock_guard<OrtMutex> lock(s_.mutex);
  ORT_RETURN_IF_ERROR(UpdateState(context));
  if (s_.Y->Shape().Size() == 0) {
    return Status::OK();
  }
  const auto alpha = Consts<CudaT>::One;
  const auto beta = Consts<CudaT>::Zero;
  IAllocatorUniquePtr<void> workspace = GetWorkSpace(context->GetComputeStream());
  auto cudnn_handle = GetCudnnHandle(context);
  CUDNN_RETURN_IF_ERROR(cudnnConvolutionForward(cudnn_handle,
                                                &alpha,
                                                s_.x_tensor,
                                                s_.x_data,
                                                s_.w_desc,
                                                s_.w_data,
                                                s_.conv_desc,
                                                s_.algo,
                                                workspace.get(),
                                                s_.workspace_bytes,
                                                &beta,
                                                s_.y_tensor,
                                                s_.y_data));
  if (nullptr != s_.b_data) {
    CUDNN_RETURN_IF_ERROR(cudnnAddTensor(cudnn_handle, &alpha, s_.b_tensor, s_.b_data,
                                         &alpha, s_.y_tensor, s_.y_data));
  }
  // To deal with asymmetric padding, we may have over-padded on one or both sides of the spatial dimensions
  // This may have lead to extra results that are unnecessary and hence we slice that off here
  if (s_.post_slicing_required) {
    ORT_RETURN_IF_ERROR(SliceOutUnwantedOutputSection(Stream(context), s_.y_data, gsl::make_span(s_.y_dims_with_adjusted_pads),
                                                      s_.Y->MutableDataRaw(), s_.y_dims.GetDims(), s_.slice_starts,
                                                      s_.slice_ends, s_.slice_axes, s_.element_size));
  }
  return Status::OK();
}



ONNX_OPERATOR_TYPED_KERNEL_EX(
    NhwcFusedConvPW,
    kMSDomain,
    1,
    float,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    NhwcFusedConvPW<float, true>);


ONNX_OPERATOR_TYPED_KERNEL_EX(
    NhwcFusedConvPW,
    kMSDomain,
    1,
    MLFloat16,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>()),
    NhwcFusedConvPW<MLFloat16, true>);


}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
