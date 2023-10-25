// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/conv_activation_fusion.h"

#include <string_view>

#include "core/common/inlined_containers.h"
#include "core/framework/tensorprotoutils.h"
#include "core/mlas/inc/mlas.h"
#include "core/graph/graph_utils.h"
#include "core/graph/node_attr_utils.h"
#include "core/optimizer/utils.h"
#include "core/optimizer/selectors_actions/actions.h"

namespace onnxruntime {

namespace {

#if !defined(ORT_MINIMAL_BUILD)
namespace selectors {
const Node* GetLoneConsumerNode(const GraphViewer& graph_viewer, const Node& node) {
  if (!optimizer_utils::CheckOutputEdges(graph_viewer.GetGraph(), node, 1)) {
    return nullptr;
  }
  return &*node.OutputNodesBegin();
}

bool HasElementDataType(const NodeArg& node_arg, int32_t data_type) {
  if (!node_arg.Exists()) {
    return false;
  }

  const auto* type_proto = node_arg.TypeAsProto();
  if (!type_proto) {
    return false;
  }

  int32_t actual_data_type;
  if (!utils::TryGetElementDataType(*type_proto, actual_data_type)) {
    return false;
  }

  return data_type == actual_data_type;
}

bool HasElementDataTypeAndAlignment(const NodeArg& node_arg, int32_t data_type, int64_t alignment) {
  if (!HasElementDataType(node_arg, data_type)) {
    return false;
  }

  const auto shape = node_arg.Shape();
  if (shape) {
    const int64_t in_channels = shape->dim(1).dim_value();
    if (in_channels % alignment) {
      return false;
    }
  } else {
    return false;
  }

  return true;
}

bool CudnnConvFusionDataTypeCheck(const Node& node) {
  // Supported dtypes: https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#conv-runtime-fusion-engine
  // ideally we can check here for alignment as well but for that we would need the current compute capability
  //
  struct cudnnAlignmentChannels {
    // int fp8 = 16; // hopper only e.g. cc90
    int64_t int8 = 4;
    int64_t fp16 = 2;
    int64_t fp32 = 1;
  };
  cudnnAlignmentChannels align;

  // #include "core/providers/cuda/shared_inc/cuda_call.h"
  // cudaDeviceProp prop;
  // CUDA_CALL_THROW(cudaGetDeviceProperties(&prop, device_id_));
  // // turing and volta
  // if (prop.major * 10 + prop.minor < 80)
  // {
  //     align.int8 = 16;
  //     align.fp16 = 8;
  //     align.fp32 = 4;
  // }

  if (!(HasElementDataTypeAndAlignment(*node.InputDefs()[0], ONNX_NAMESPACE::TensorProto_DataType_FLOAT, align.fp32) ||
        HasElementDataTypeAndAlignment(*node.InputDefs()[0], ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16, align.fp16) ||
        HasElementDataTypeAndAlignment(*node.InputDefs()[0], ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, align.fp16) ||
        // TODO unsure about fp8 support in ONNX
        // HasElementDataType(*node.InputDefs()[0], ONNX_NAMESPACE::TensorProto_DataType_FLOAT8) ||
        HasElementDataTypeAndAlignment(*node.InputDefs()[0], ONNX_NAMESPACE::TensorProto_DataType_INT8, align.int8))) {
    return false;
  }
  return true;
}

bool ConvFusionDataTypeCheck(const Node& conv_node) {
  // TODO(hasesh): The CPU and CUDA EP only support float type for the Conv+Activation
  // and the Conv+Add+Relu fusions.
  // Assess the support level for the other compatible EPs and if they also
  // only support float, remove the EP check altogether.
  const std::string_view node_ep = conv_node.GetExecutionProviderType();
  if (node_ep == kCudaExecutionProvider) {
    if (!HasElementDataType(*conv_node.InputDefs()[0], ONNX_NAMESPACE::TensorProto_DataType_FLOAT)) {
      return false;
    }
  }
  if (node_ep == kCpuExecutionProvider) {
#ifdef MLAS_F16VEC_INTRINSICS_SUPPORTED
    if (!HasElementDataType(*conv_node.InputDefs()[0], ONNX_NAMESPACE::TensorProto_DataType_FLOAT) &&
        !HasElementDataType(*conv_node.InputDefs()[0], ONNX_NAMESPACE::TensorProto_DataType_FLOAT16)) {
      return false;
    }
#else
    if (!HasElementDataType(*conv_node.InputDefs()[0], ONNX_NAMESPACE::TensorProto_DataType_FLOAT)) {
      return false;
    }
#endif  // MLAS_F16VEC_INTRINSICS_SUPPORTED
  }

  return true;
}

static bool IsPWMathNode(const Node* node) {
  if (!node) {
    return false;
  }
  if (node->GetOutputEdgesCount() > 1) {
    return false;
  }

  return graph_utils::IsSupportedOptypeVersionAndDomain(*node, "Add", {6, 7, 13, 14}) ||
         graph_utils::IsSupportedOptypeVersionAndDomain(*node, "Mul", {12, 13, 14});
}

static bool IsActivationNode(const Node* node) {
  if (!node) {
    return false;
  }
  if (node->GetOutputEdgesCount() > 1) {
    return false;
  }

  return graph_utils::IsSupportedOptypeVersionAndDomain(*node, "Relu", {6, 13, 14}) ||
         graph_utils::IsSupportedOptypeVersionAndDomain(*node, "Sigmoid", {6, 13}) ||
        //  graph_utils::IsSupportedOptypeVersionAndDomain(*node, "LeakyRelu", {6, 16}) ||
         graph_utils::IsSupportedOptypeVersionAndDomain(*node, "Tanh", {6, 13});
}

bool IsPointwiseNode(const Node* node) {
  if (!node) {
    return false;
  }
  if (node->GetOutputEdgesCount() > 1) {
    return false;
  }

  return IsActivationNode(node) || IsPWMathNode(node);
}

class ConvActivationSelector : public NodeSelector {
 public:
  ConvActivationSelector() = default;

  std::optional<NodesToOptimizeIndices> Select(const GraphViewer& graph_viewer, const Node& node) const override {
    const std::string_view node_ep = node.GetExecutionProviderType();
    const auto* next_node = GetLoneConsumerNode(graph_viewer, node);
    if (!next_node ||
        next_node->GetExecutionProviderType() != node_ep) {
      return std::nullopt;
    }

    auto is_supported_non_cuda_rocm_ep_activation = [&graph_viewer](const Node& activation_node) {
      if (graph_utils::IsSupportedOptypeVersionAndDomain(activation_node, "Relu", {6, 13, 14}) ||
          graph_utils::IsSupportedOptypeVersionAndDomain(activation_node, "Sigmoid", {6, 13}) ||
          graph_utils::IsSupportedOptypeVersionAndDomain(activation_node, "Tanh", {6, 13}) ||
          graph_utils::IsSupportedOptypeVersionAndDomain(activation_node, "LeakyRelu", {6, 16})) {
        return true;
      }

      if (graph_utils::IsSupportedOptypeVersionAndDomain(activation_node, "Clip", {6, 11, 12, 13})) {
        float min, max;
        if (!optimizer_utils::GetClipConstantMinMax(graph_viewer.GetGraph(), activation_node, min, max)) {
          return false;
        }
        return true;
      }

      return false;
    };

    if (!ConvFusionDataTypeCheck(node)) {
      return std::nullopt;
    }

    // check EP type and activation
    if (node_ep == kCudaExecutionProvider) {
      if (!IsActivationNode(next_node)) {
        return std::nullopt;
      }
    } else if (node_ep == kRocmExecutionProvider) {
      if (!graph_utils::IsSupportedOptypeVersionAndDomain(*next_node, "Relu", {6, 13, 14})) {
        return std::nullopt;
      }
    } else if (node_ep.empty() || node_ep == kCpuExecutionProvider) {
      if (!is_supported_non_cuda_rocm_ep_activation(*next_node) &&
          !graph_utils::IsSupportedOptypeVersionAndDomain(*next_node, "HardSigmoid", {6})) {
        return std::nullopt;
      }
    } else {
      if (!is_supported_non_cuda_rocm_ep_activation(*next_node)) {
        return std::nullopt;
      }
    }

    NodesToOptimizeIndicesBuilder builder{};
    builder.target_node = node.Index();
    builder.output_nodes.push_back(next_node->Index());
    return builder.Build();
  }
};

class ConvAddRelu : public NodeSelector {
 public:
  ConvAddRelu() = default;

  std::optional<NodesToOptimizeIndices> Select(const GraphViewer& graph_viewer, const Node& node) const override {
    const std::string_view node_ep = node.GetExecutionProviderType();
    // only for CUDA EP
    if (node_ep != kCudaExecutionProvider) {
      return std::nullopt;
    }

    if (!ConvFusionDataTypeCheck(node)) {
      return std::nullopt;
    }

    const auto* add_node = GetLoneConsumerNode(graph_viewer, node);
    if (!add_node ||
        !graph_utils::IsSupportedOptypeVersionAndDomain(*add_node, "Add", {6, 7, 13, 14}) ||
        add_node->GetExecutionProviderType() != node_ep) {
      return std::nullopt;
    }

    const auto* relu_node = GetLoneConsumerNode(graph_viewer, *add_node);
    if (!relu_node ||
        !graph_utils::IsSupportedOptypeVersionAndDomain(*relu_node, "Relu", {6, 13, 14}) ||
        relu_node->GetExecutionProviderType() != node_ep) {
      return std::nullopt;
    }

    NodesToOptimizeIndicesBuilder builder{};
    builder.target_node = node.Index();
    builder.output_nodes = {add_node->Index(),
                            relu_node->Index()};
    return builder.Build();
  }
};

class ConvPW : public NodeSelector {
 public:
  ConvPW() = default;

  std::optional<NodesToOptimizeIndices> Select(const GraphViewer& graph_viewer, const Node& node) const override {
    const std::string_view node_ep = node.GetExecutionProviderType();
    // only for CUDA EP
    if (node_ep != kCudaExecutionProvider) {
      return std::nullopt;
    }

    if (!CudnnConvFusionDataTypeCheck(node)) {
      return std::nullopt;
    }

    NodesToOptimizeIndicesBuilder builder{};
    builder.target_node = node.Index();
    auto* pw_node = GetLoneConsumerNode(graph_viewer, node);
    // TODO here it would be much nicer to check if the next pointwise node is supported by cudnn frontend
    while (IsPointwiseNode(pw_node)) {
      if (pw_node->GetExecutionProviderType() != kCudaExecutionProvider) {
        break;
      }

      builder.output_nodes.push_back(pw_node->Index());
      pw_node = GetLoneConsumerNode(graph_viewer, *pw_node);
    }
    if (!builder.output_nodes.size()) {
      return std::nullopt;
    }
    return builder.Build();
  }
};

}  // namespace selectors
#endif  // !defined(ORT_MINIMAL_BUILD)

namespace actions {
using NTO = NodesToOptimize;

class FuseConvActivationAction : public ReplaceWithNew {
 private:
  std::string OpType(const RuntimeState&) const override { return "FusedConv"; }

  std::string Domain(const RuntimeState&) const override { return kMSDomain; }

  NodeAttributes ExtraAttributes(const RuntimeState& state) const override {
    NodeAttributes extra_fused_conv_attributes;

    const auto* activation = state.selected_nodes.Output(0);
    ORT_ENFORCE(activation != nullptr, "Expected activation node.");

    const auto& activation_op_type = activation->OpType();
    utils::SetNodeAttribute(utils::MakeAttribute("activation", activation_op_type), extra_fused_conv_attributes);

    InlinedVector<float> activation_params;
    if (activation_op_type == "LeakyRelu") {
      activation_params.push_back(graph_utils::GetNodeAttribute(*activation, "alpha")->f());
    } else if (activation_op_type == "Clip") {
      float min, max;
      ORT_ENFORCE(optimizer_utils::GetClipConstantMinMax(state.graph, *activation, min, max),
                  "Failed to get Clip min/max constants.");
      activation_params.push_back(min);
      activation_params.push_back(max);
    } else if (activation_op_type == "HardSigmoid") {
      auto* alpha_attr = graph_utils::GetNodeAttribute(*activation, "alpha");
      auto* beta_attr = graph_utils::GetNodeAttribute(*activation, "beta");
      float alpha = (alpha_attr == nullptr ? 0.2f : alpha_attr->f());
      float beta = (beta_attr == nullptr ? 0.5f : beta_attr->f());
      activation_params.push_back(alpha);
      activation_params.push_back(beta);
    }

    if (!activation_params.empty()) {
      utils::SetNodeAttribute(utils::MakeAttribute("activation_params", activation_params),
                              extra_fused_conv_attributes);
    }

    return extra_fused_conv_attributes;
  }

  std::vector<NodeAndMoveInfo> ValueMoves(const RuntimeState&) const override {
    const NTO::NodeLocation conv{NTO::NodeType::kTarget, 0};
    const NTO::NodeLocation activation{NTO::NodeType::kOutput, 0};

    return {
        MoveAll(conv, ArgType::kInput),         // move all inputs from conv
        MoveAll(activation, ArgType::kOutput),  // move all outputs from activation
    };
  }
};

class FuseConvAddRelu : public ReplaceWithNew {
 private:
  std::string OpType(const RuntimeState&) const override { return "FusedConv"; }

  std::string Domain(const RuntimeState&) const override { return kMSDomain; }

  NodeAttributes ExtraAttributes(const RuntimeState&) const override {
    NodeAttributes extra_fused_conv_attributes;
    utils::SetNodeAttribute(utils::MakeAttribute("activation", "Relu"), extra_fused_conv_attributes);
    return extra_fused_conv_attributes;
  }

  std::vector<NodeAndMoveInfo> ValueMoves(const RuntimeState& state) const override {
    const auto& conv = state.selected_nodes.Target();

    ORT_ENFORCE(conv.GetOutputEdgesCount() == 1 && conv.OutputNodesBegin()->OpType() == "Add",
                "Expected Conv then Add.");
    const auto add_input_idx = 1 - conv.OutputEdgesBegin()->GetDstArgIndex();

    const auto conv_location = NTO::NodeLocation{NTO::NodeType::kTarget, 0};
    const auto add_location = NTO::NodeLocation{NTO::NodeType::kOutput, 0};
    const auto relu_location = NTO::NodeLocation{NTO::NodeType::kOutput, 1};

    return {
        MoveAll(conv_location, ArgType::kInput),                                       // move all inputs from conv
        MoveAndAppend(add_location, ArgType::kInput, add_input_idx, ArgType::kInput),  // append add input
        MoveAll(relu_location, ArgType::kOutput),                                      // move all outputs from relu
    };
  }
};

class FuseConvPointwiseAction : public ReplaceWithNew {
 private:
  std::string OpType(const RuntimeState&) const override { return "NhwcFusedConvPW"; }

  std::string Domain(const RuntimeState&) const override { return kMSDomain; }

  NodeAttributes ExtraAttributes(const RuntimeState& state) const override {
    NodeAttributes extra_fused_conv_attributes;
    for (int nto_idx = 0; nto_idx < state.selected_nodes.num_outputs; ++nto_idx) {
      const auto* pw_node = state.selected_nodes.Output(nto_idx);
      if (selectors::IsActivationNode(pw_node)) {
        const auto& activation_op_type = pw_node->OpType();
        utils::SetNodeAttribute(utils::MakeAttribute("activation", activation_op_type), extra_fused_conv_attributes);

        InlinedVector<float> activation_params;
        if (activation_op_type == "LeakyRelu") {
          activation_params.push_back(graph_utils::GetNodeAttribute(*pw_node, "alpha")->f());
        }

        if (!activation_params.empty()) {
          utils::SetNodeAttribute(utils::MakeAttribute("activation_params", activation_params),
                                  extra_fused_conv_attributes);
        }
      }
    }
    return extra_fused_conv_attributes;
  }

  std::vector<NodeAndMoveInfo> ValueMoves(const RuntimeState& state) const override {
    const auto& conv = state.selected_nodes.Target();
    ORT_ENFORCE(conv.GetOutputEdgesCount() == 1, "Expected nodes with single output");

    const auto conv_location = NTO::NodeLocation{NTO::NodeType::kTarget, 0};
    std::vector<NodeAndMoveInfo> node_move_info = {
        MoveAll(conv_location, ArgType::kInput),
        // MoveAndAppend(NTO::NodeLocation{NTO::NodeType::kInput, 2}, ArgType::kInput, 0, ArgType::kInput,
        //               true, true) // fill optional values
    };
    bool add_used = false, mul_used = false;
    for (int nto_idx = 0; nto_idx < state.selected_nodes.num_outputs; ++nto_idx) {
      const auto* pw_node = state.selected_nodes.Output(nto_idx);
      if (nto_idx != (state.selected_nodes.num_outputs - 1)) {
        ORT_ENFORCE(pw_node->GetOutputEdgesCount() == 1, "Expected nodes with single output");
        // TODO the check against == 1 fails if it is the last node in the network
      }

      // TODO find out how to first fill and then move to slot like here ?
      // https://github.com/gedoensmax/onnxruntime/blob/948ea37ef7ffae91e5c2c008a37aacb386f3c0b1/onnxruntime/core/optimizer/qdq_transformer/selectors_actions/qdq_actions.cc
      if (graph_utils::IsSupportedOptypeVersionAndDomain(*pw_node, "Add", {6, 7, 13, 14})) {
        ORT_ENFORCE(!add_used, "Fusing multiple adds is not possible");
        const auto add_location = NTO::NodeLocation{NTO::NodeType::kOutput, nto_idx};
        // const auto input_idx = 1 - pw_node->OutputEdgesBegin()->GetDstArgIndex();
        // node_move_info.push_back(MoveToSlot(add_location, ArgType::kInput, 1, ArgType::kInput, 4));
        node_move_info.push_back(MoveAndAppend(add_location, ArgType::kInput, 1, ArgType::kInput));
        add_used = true;
      } else if (graph_utils::IsSupportedOptypeVersionAndDomain(*pw_node, "Mul", {12, 13, 14})) {
        ORT_ENFORCE(!mul_used, "Fusing multiple adds is not possible");
        const auto mul_location = NTO::NodeLocation{NTO::NodeType::kOutput, nto_idx};
        // const auto input_idx = 1 - pw_node->OutputEdgesBegin()->GetDstArgIndex();
        // node_move_info.push_back(MoveToSlot(mul_location, ArgType::kInput, 1, ArgType::kInput));
        node_move_info.push_back(MoveAndAppend(mul_location, ArgType::kInput, 1, ArgType::kInput));
        mul_used = true;
      }
    }
    const auto out_location = NTO::NodeLocation{NTO::NodeType::kOutput, state.selected_nodes.num_outputs - 1};
    node_move_info.push_back(MoveAll(out_location, ArgType::kOutput));

    return node_move_info;
  }
};
}  // namespace actions

// void RegisterConvPointwiseFusionRules(SelectorActionRegistry& registry) {
//   const auto name = "ConvPW";
//   auto action = std::make_unique<actions::FuseConvPointwiseAction>();
// #if !defined(ORT_MINIMAL_BUILD)
//   auto selector = std::make_unique<selectors::ConvPW>();
//   registry.RegisterSelectorAndAction(name, {{"Conv", {1, 11}}},
//                                      std::move(selector), std::move(action));
// #else
//   registry.RegisterAction(name, std::move(action));
// #endif
// }

void RegisterConvActivationFusionRules(SelectorActionRegistry& registry) {
  const auto name = "ConvAct";
  auto action = std::make_unique<actions::FuseConvActivationAction>();
#if !defined(ORT_MINIMAL_BUILD)
  auto selector = std::make_unique<selectors::ConvActivationSelector>();
  registry.RegisterSelectorAndAction(name, {{"Conv", {1, 11}}},
                                     std::move(selector), std::move(action));
#else
  registry.RegisterAction(name, std::move(action));
#endif
}

void RegisterConvAddReluFusionRules(SelectorActionRegistry& registry) {
  const auto name = "ConvAddRelu";
  auto action = std::make_unique<actions::FuseConvAddRelu>();
#if !defined(ORT_MINIMAL_BUILD)
  auto selector = std::make_unique<selectors::ConvAddRelu>();
  registry.RegisterSelectorAndAction(name, {{"Conv", {1, 11}}},
                                     std::move(selector), std::move(action));
#else
  registry.RegisterAction(name, std::move(action));
#endif
}

SelectorActionRegistry CreateSelectorActionRegistry() {
  SelectorActionRegistry registry{};
  // RegisterConvPointwiseFusionRules(registry);
  RegisterConvActivationFusionRules(registry);
  RegisterConvAddReluFusionRules(registry);
  return registry;
}

}  // namespace

ConvActivationFusion::ConvActivationFusion(const InlinedHashSet<std::string_view>& compatible_execution_providers,
                                           const SatApplyContextVariant& apply_context)
    : SelectorActionTransformer{
          "ConvActivationFusion", CreateSelectorActionRegistry(), apply_context, compatible_execution_providers} {
}

}  // namespace onnxruntime
