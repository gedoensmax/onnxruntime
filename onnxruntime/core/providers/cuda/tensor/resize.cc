// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "resize.h"

namespace onnxruntime {
namespace cuda {
#define REGISTER_KERNEL_TYPED(T, NHWC, DOMAIN)                     \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                         \
      Resize,                                                      \
      DOMAIN,                                                      \
      10, 10,                                                      \
      T,                                                           \
      kCudaExecutionProvider,                                      \
      (*KernelDefBuilder::Create())                                \
          .InputMemoryType(OrtMemTypeCPUInput, 1)                  \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),  \
      Resize<T, NHWC>);                                            \
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
      Resize<T, NHWC>);                                            \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                   \
      Resize,                                                      \
      DOMAIN,                                                      \
      13,                                                          \
      T,                                                           \
      kCudaExecutionProvider,                                      \
      (*KernelDefBuilder::Create())                                \
          .InputMemoryType(OrtMemTypeCPUInput, 1)                  \
          .InputMemoryType(OrtMemTypeCPUInput, 2)                  \
          .InputMemoryType(OrtMemTypeCPUInput, 3)                  \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>()), \
      Resize<T, NHWC>);                                            \
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
      Resize<T, NHWC>);
// TODO: opset 18 is not verified and was just added as a dummy for now !


REGISTER_KERNEL_TYPED(float, false, kOnnxDomain)
REGISTER_KERNEL_TYPED(double, false, kOnnxDomain)
REGISTER_KERNEL_TYPED(MLFloat16, false, kOnnxDomain)
REGISTER_KERNEL_TYPED(int32_t, false, kOnnxDomain)
REGISTER_KERNEL_TYPED(uint8_t, false, kOnnxDomain)

REGISTER_KERNEL_TYPED(float, true, kMSInternalNHWCDomain)
REGISTER_KERNEL_TYPED(double, true, kMSInternalNHWCDomain)
REGISTER_KERNEL_TYPED(MLFloat16, true, kMSInternalNHWCDomain)
REGISTER_KERNEL_TYPED(int32_t, true, kMSInternalNHWCDomain)
REGISTER_KERNEL_TYPED(uint8_t, true, kMSInternalNHWCDomain)

}  // namespace cuda
}  // namespace onnxruntime
