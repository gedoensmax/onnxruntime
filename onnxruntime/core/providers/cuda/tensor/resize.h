// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/tensor/upsample.h"

namespace onnxruntime {
namespace cuda {

template <typename T, bool NHWC>
class Resize : public Upsample<T, NHWC> {
 public:
  Resize(const OpKernelInfo& info) : Upsample<T, NHWC>(info) {
  }

  Status ComputeInternal(OpKernelContext* context) const override {
    return Upsample<T, NHWC>::ComputeInternal(context);
  }
};

}  // namespace cuda
}  // namespace onnxruntime
