// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "resize.h"

namespace onnxruntime {
namespace cuda {
#define REGISTER_KERNEL_TYPED(T, DOMAIN)                           \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                         \
      Resize,                                                      \
      DOMAIN,                                                      \
      10, 10,                                                      \
      T,                                                           \
      kCudaExecutionProvider,                                      \
      (*KernelDefBuilder::Create())                                \
          .InputMemoryType(OrtMemTypeCPUInput, 1)                  \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),  \
      Resize<T>);                                                  \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                         \
      Resize,                                                      \
      DOMAIN,                                                      \
      11, 12,                                                      \
      T,                                                           \
      kCudaExecutionProvider,                                      \
      (*KernelDefBuilder::Create())                                \
          .InputMemoryType(OrtMemTypeCPUInput, 1)                  \
          .InputMemoryType(OrtMemTypeCPUInput, 2)                  \
          .InputMemoryType(OrtMemTypeCPUInput, 3)                  \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>()), \
      Resize<T>);                                                  \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                         \
      Resize,                                                      \
      DOMAIN,                                                      \
      13, 17,                                                      \
      T,                                                           \
      kCudaExecutionProvider,                                      \
      (*KernelDefBuilder::Create())                                \
          .InputMemoryType(OrtMemTypeCPUInput, 1)                  \
          .InputMemoryType(OrtMemTypeCPUInput, 2)                  \
          .InputMemoryType(OrtMemTypeCPUInput, 3)                  \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>()), \
      Resize<T>);                                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                   \
      Resize,                                                      \
      DOMAIN,                                                      \
      18,                                                          \
      T,                                                           \
      kCudaExecutionProvider,                                      \
      (*KernelDefBuilder::Create())                                \
          .InputMemoryType(OrtMemTypeCPUInput, 1)                  \
          .InputMemoryType(OrtMemTypeCPUInput, 2)                  \
          .InputMemoryType(OrtMemTypeCPUInput, 3)                  \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>()), \
      Resize<T>);

REGISTER_KERNEL_TYPED(float, kOnnxDomain)
REGISTER_KERNEL_TYPED(double, kOnnxDomain)
REGISTER_KERNEL_TYPED(MLFloat16, kOnnxDomain)
REGISTER_KERNEL_TYPED(int32_t, kOnnxDomain)
REGISTER_KERNEL_TYPED(int8_t, kOnnxDomain)
REGISTER_KERNEL_TYPED(uint8_t, kOnnxDomain)

#ifdef ENABLE_CUDA_NHWC_OPS
REGISTER_KERNEL_TYPED(float, kMSInternalNHWCDomain)
REGISTER_KERNEL_TYPED(double, kMSInternalNHWCDomain)
REGISTER_KERNEL_TYPED(MLFloat16, kMSInternalNHWCDomain)
REGISTER_KERNEL_TYPED(int32_t, kMSInternalNHWCDomain)
REGISTER_KERNEL_TYPED(int8_t, kMSInternalNHWCDomain)
REGISTER_KERNEL_TYPED(uint8_t, kMSInternalNHWCDomain)
#endif

}  // namespace cuda
}  // namespace onnxruntime
