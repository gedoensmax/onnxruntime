// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2023 NVIDIA Corporation.
// Licensed under the MIT License.

#include "test/providers/cuda/nhwc/nhwc_cuda_helper.h"

namespace onnxruntime {
namespace test {

template <typename T>
struct GridSampleOp {
  const std::vector<int64_t> input_dims;
  const std::vector<int64_t> grid_dims;
  const std::string padding_mode;

  std::unique_ptr<CompareOpTester> get_test() {
    RandomValueGenerator random{};

    auto test = std::make_unique<CompareOpTester>("GridSample", 16);
    std::vector<T> input_data = random.Uniform<T>(input_dims, 0.0f, 0.3f);

    test->AddInput<T>("X", input_dims, input_data);

    std::vector<T> grid = random.Uniform<T>(grid_dims, -1.1f, 1.1f);
    test->AddInput<T>("grid", grid_dims, grid);
    if (!padding_mode.empty()) {
      test->AddAttribute("padding_mode", padding_mode);
    }
    std::vector<T> output_data = FillZeros<T>(grid_dims);
    test->AddOutput<T>("Y", grid_dims, output_data);
    return test;
  }
};

TYPED_TEST(CudaNhwcTypedTest, GridSampleNhwcSmall) {
  auto op = GridSampleOp<TypeParam>{
      .input_dims = {1, 8, 64, 64},
      .grid_dims = {1, 8, 8, 2}};
  MAKE_PROVIDERS()
}

TYPED_TEST(CudaNhwcTypedTest, GridSampleNhwcBig) {
  auto op = GridSampleOp<TypeParam>{
      .input_dims = {1, 4, 64, 64},
      .grid_dims = {1, 112, 112, 2}};
  MAKE_PROVIDERS()
}

TYPED_TEST(CudaNhwcTypedTest, GridSampleNhwcPadding) {
  auto op = GridSampleOp<TypeParam>{
      .input_dims = {1, 4, 64, 64},
      .grid_dims = {1, 112, 112, 2},
      .padding_mode = "reflection"};

  MAKE_PROVIDERS()
}

}  // namespace test
}  // namespace onnxruntime
