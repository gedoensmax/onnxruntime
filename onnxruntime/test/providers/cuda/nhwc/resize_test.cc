// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2023 NVIDIA Corporation.
// Licensed under the MIT License.

#include "test/providers/cuda/nhwc/nhwc_cuda_helper.h"

namespace onnxruntime {
namespace test {

template <typename T>
struct ResizeOp {
  const std::vector<int64_t> input_dims;
  std::vector<float> scales;
  std::vector<int64_t> sizes;
  std::vector<int64_t> output_dims;

  std::unique_ptr<CompareOpTester> get_test() {
    RandomValueGenerator random{};

    auto test = std::make_unique<CompareOpTester>("Resize", 13);
    std::vector<T> input_data = random.Uniform<T>(input_dims, 0.0f, 0.3f);

    test->AddInput<T>("X", input_dims, input_data);

    std::vector<int64_t> dims = {static_cast<int64_t>(input_dims.size())};
    test->AddInput<float>("roi", {0}, {});
    if (!scales.empty()) {
      test->AddInput<float>("scales", dims, scales);
      for (size_t i = 0; i < input_dims.size(); ++i) {
        output_dims.push_back(input_dims[i] * scales[i]);
      }
    }
    if (!sizes.empty()) {
      test->AddInput<int64_t >("sizes", dims, sizes);
      output_dims = sizes;
    }
    std::vector<T> output_data = FillZeros<T>(output_dims);
    test->AddOutput<T>("Y", output_dims, output_data);
    return test;
  }
};

TYPED_TEST(CudaNhwcTypedTest, ResizeNhwcScales) {
  {
    auto op = ResizeOp<TypeParam>{
        .input_dims = {1, 8, 64, 64},
        .scales = {1, 1, 2, 2}};
    MAKE_PROVIDERS()
  }
  {
    auto op = ResizeOp<TypeParam>{
        .input_dims = {1, 8, 64, 64},
        .scales = {1, 1, 0.3f, 0.3f}};
    MAKE_PROVIDERS()
  }
}

TYPED_TEST(CudaNhwcTypedTest, ResizeNhwcSizes) {
  {
    auto op = ResizeOp<TypeParam>{
        .input_dims = {1, 8, 64, 64},
        .sizes = {1, 8, 16, 16}};
    MAKE_PROVIDERS()
  }
  {
    auto op = ResizeOp<TypeParam>{
        .input_dims = {1, 8, 64, 64},
        .sizes = {1, 16, 128, 128}};
    MAKE_PROVIDERS()
  }
}

}  // namespace test
}  // namespace onnxruntime
