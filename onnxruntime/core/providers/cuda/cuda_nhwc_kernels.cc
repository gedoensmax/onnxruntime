// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2023 NVIDIA Corporation.
// Licensed under the MIT License.

#ifdef ENABLE_CUDA_NHWC_OPS

#include <utility>

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/cuda_fwd.h"

#include "core/providers/cuda/cuda_nhwc_kernels.h"

namespace onnxruntime::cuda {

// When adding new supported NHWC operations make sure to also integrate them into: ConvertNodeLayout
// in onnxruntime/core/optimizer/layout_transformation/layout_transformation.cc

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 7, 8, float,
                                                      BatchNormalization);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 7, 8, MLFloat16,
                                                      BatchNormalization);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 9, 13, float,
                                                      BatchNormalization);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 9, 13, MLFloat16,
                                                      BatchNormalization);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 1, 10, float,
                                                      Conv);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 1, 10, MLFloat16,
                                                      Conv);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 1, 10, float,
                                                      ConvTranspose);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 1, 10, MLFloat16,
                                                      ConvTranspose);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 7, 9, float,
                                                      AveragePool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 7, 9, MLFloat16,
                                                      AveragePool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 1, float, GlobalAveragePool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 1, MLFloat16,
                                            GlobalAveragePool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 1, 7, float,
                                                      MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 1, 7, MLFloat16,
                                                      MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 8, 9, float,
                                                      MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 8, 9, MLFloat16,
                                                      MaxPool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 1, float, GlobalMaxPool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 1, MLFloat16, GlobalMaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 10, 10, float,
                                                      AveragePool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 10, 10, MLFloat16,
                                                      AveragePool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 10, 10, float,
                                                      MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 10, 10, MLFloat16,
                                                      MaxPool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 11, float, Conv);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 11, MLFloat16, Conv);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 11, float, ConvTranspose);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 11, MLFloat16,
                                            ConvTranspose);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 11, float, AveragePool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 11, MLFloat16, AveragePool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 11, 11, float,
                                                      MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 11, 11, MLFloat16,
                                                      MaxPool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 12, float, MaxPool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 12, MLFloat16, MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 14, 14, float,
                                                      BatchNormalization);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 14, 14, MLFloat16,
                                                      BatchNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 15, float,
                                            BatchNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 15, MLFloat16,
                                            BatchNormalization);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 1, 10, DepthToSpace);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 1, 12, SpaceToDepth);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 11, 12, DepthToSpace);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 13, SpaceToDepth);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 13, DepthToSpace);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 7, 8, float,
                                                      Upsample);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 7, 8, double,
                                                      Upsample);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 7, 8, MLFloat16,
                                                      Upsample);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 7, 8, int32_t,
                                                      Upsample);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 7, 8, uint8_t,
                                                      Upsample);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 9, 9, float,
                                                      Upsample);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 9, 9, double,
                                                      Upsample);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 9, 9, MLFloat16,
                                                      Upsample);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 9, 9, int32_t,
                                                      Upsample);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 9, 9, uint8_t,
                                                      Upsample);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 10, 10, float,
                                                      Resize);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 10, 10, double,
                                                      Resize);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 10, 10, MLFloat16,
                                                      Resize);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 10, 10, int32_t,
                                                      Resize);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 10, 10, uint8_t,
                                                      Resize);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 11, 12, float,
                                                      Resize);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 11, 12, double,
                                                      Resize);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 11, 12, MLFloat16,
                                                      Resize);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 11, 12, int32_t,
                                                      Resize);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 11, 12, uint8_t,
                                                      Resize);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 13, float, Resize);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 13, double, Resize);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 13, MLFloat16, Resize);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 13, int32_t, Resize);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 13, uint8_t, Resize);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 18, float, Resize);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 18, double, Resize);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 18, MLFloat16, Resize);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 18, int32_t, Resize);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 18, uint8_t, Resize);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 13, float, LRN);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 13, double, LRN);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 13, MLFloat16, LRN);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 1, 12, float, LRN);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 1, 12, double,
                                                      LRN);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 1, 12, MLFloat16,
                                                      LRN);

Status RegisterCudaNhwcKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn nhwc_function_table[] = {
      BuildKernelCreateInfo<void>,  // default entry to avoid the list become empty after ops-reducing
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kCudaExecutionProvider, kMSInternalNHWCDomain, 7, 8, MLFloat16, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kCudaExecutionProvider, kMSInternalNHWCDomain, 7, 8, float, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kCudaExecutionProvider, kMSInternalNHWCDomain, 9, 13, MLFloat16, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kCudaExecutionProvider, kMSInternalNHWCDomain, 9, 13, float, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kCudaExecutionProvider, kMSInternalNHWCDomain, 14, 14, MLFloat16, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kCudaExecutionProvider, kMSInternalNHWCDomain, 14, 14, float, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 15,
                                                                  MLFloat16, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 15,
                                                                  float, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kCudaExecutionProvider, kMSInternalNHWCDomain, 1, 10, MLFloat16, Conv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider,
                                                                            kMSInternalNHWCDomain, 1, 10, float, Conv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 11,
                                                                  float, Conv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 11,
                                                                  MLFloat16, Conv)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kCudaExecutionProvider, kMSInternalNHWCDomain, 7, 9, float, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kCudaExecutionProvider, kMSInternalNHWCDomain, 7, 9, MLFloat16, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 1,
                                                                  float, GlobalAveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 1,
                                                                  MLFloat16, GlobalAveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kCudaExecutionProvider, kMSInternalNHWCDomain, 1, 7, float, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kCudaExecutionProvider, kMSInternalNHWCDomain, 1, 7, MLFloat16, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kCudaExecutionProvider, kMSInternalNHWCDomain, 8, 9, float, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kCudaExecutionProvider, kMSInternalNHWCDomain, 8, 9, MLFloat16, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 1,
                                                                  float, GlobalMaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 1,
                                                                  MLFloat16, GlobalMaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kCudaExecutionProvider, kMSInternalNHWCDomain, 10, 10, float, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kCudaExecutionProvider, kMSInternalNHWCDomain, 10, 10, MLFloat16, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kCudaExecutionProvider, kMSInternalNHWCDomain, 10, 10, float, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kCudaExecutionProvider, kMSInternalNHWCDomain, 10, 10, MLFloat16, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 11,
                                                                  float, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 11,
                                                                  MLFloat16, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kCudaExecutionProvider, kMSInternalNHWCDomain, 11, 11, float, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kCudaExecutionProvider, kMSInternalNHWCDomain, 11, 11, MLFloat16, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 12,
                                                                  float, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 12,
                                                                  MLFloat16, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 11,
                                                                  float, ConvTranspose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 11,
                                                                  MLFloat16, ConvTranspose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kCudaExecutionProvider, kMSInternalNHWCDomain, 1, 10, float, ConvTranspose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kCudaExecutionProvider, kMSInternalNHWCDomain, 1, 10, MLFloat16, ConvTranspose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 1,
                                                                      10, DepthToSpace)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 1,
                                                                      12, SpaceToDepth)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 11,
                                                                      12, DepthToSpace)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 13,
                                                            SpaceToDepth)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 13,
                                                            DepthToSpace)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kCudaExecutionProvider, kMSInternalNHWCDomain, 7, 8, float, Upsample)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kCudaExecutionProvider, kMSInternalNHWCDomain, 7, 8, double, Upsample)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kCudaExecutionProvider, kMSInternalNHWCDomain, 7, 8, MLFloat16, Upsample)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kCudaExecutionProvider, kMSInternalNHWCDomain, 7, 8, int32_t, Upsample)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kCudaExecutionProvider, kMSInternalNHWCDomain, 7, 8, uint8_t, Upsample)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kCudaExecutionProvider, kMSInternalNHWCDomain, 9, 9, float, Upsample)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kCudaExecutionProvider, kMSInternalNHWCDomain, 9, 9, double, Upsample)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kCudaExecutionProvider, kMSInternalNHWCDomain, 9, 9, MLFloat16, Upsample)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kCudaExecutionProvider, kMSInternalNHWCDomain, 9, 9, int32_t, Upsample)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kCudaExecutionProvider, kMSInternalNHWCDomain, 9, 9, uint8_t, Upsample)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kCudaExecutionProvider, kMSInternalNHWCDomain, 10, 10, float, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kCudaExecutionProvider, kMSInternalNHWCDomain, 10, 10, double, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kCudaExecutionProvider, kMSInternalNHWCDomain, 10, 10, MLFloat16, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kCudaExecutionProvider, kMSInternalNHWCDomain, 10, 10, int32_t, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kCudaExecutionProvider, kMSInternalNHWCDomain, 10, 10, uint8_t, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kCudaExecutionProvider, kMSInternalNHWCDomain, 11, 12, float, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kCudaExecutionProvider, kMSInternalNHWCDomain, 11, 12, double, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kCudaExecutionProvider, kMSInternalNHWCDomain, 11, 12, MLFloat16, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kCudaExecutionProvider, kMSInternalNHWCDomain, 11, 12, int32_t, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kCudaExecutionProvider, kMSInternalNHWCDomain, 11, 12, uint8_t, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 13,
                                                                  float, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 13,
                                                                  double, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 13,
                                                                  MLFloat16, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 13,
                                                                  int32_t, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 13,
                                                                  uint8_t, Resize)>,
      // TODO opset 18 unverified ! Just added to make it "run" as the NHWC transformer seems to ingest a newer opset
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 18,
                                                                  float, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 18,
                                                                  double, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 18,
                                                                  MLFloat16, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 18,
                                                                  int32_t, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 18,
                                                                  uint8_t, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider,
                                                                            kMSInternalNHWCDomain, 1, 12, float, LRN)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider,
                                                                            kMSInternalNHWCDomain, 1, 12, double, LRN)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kCudaExecutionProvider, kMSInternalNHWCDomain, 1, 12, MLFloat16, LRN)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 13,
                                                                  float, LRN)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 13,
                                                                  double, LRN)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 13,
                                                                  MLFloat16, LRN)>,
  };

  for (auto& function_table_entry : nhwc_function_table) {
    KernelCreateInfo info = function_table_entry();
    if (info.kernel_def != nullptr) {  // filter disabled entries where type is void
      ORT_RETURN_IF_ERROR(kernel_registry.Register(std::move(info)));
    }
  }
  return Status::OK();
}
}  // namespace onnxruntime::cuda
#endif
