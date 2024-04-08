// Copyright (c) NVIDIA Corporation. All rights reserved.
// Licensed under the MIT License.


#include "core/optimizer/group_norm_detection.h"

#include "core/graph/graph_utils.h"
#include "core/graph/node_attr_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"
#include "core/optimizer/selectors_actions/actions.h"


using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;
namespace onnxruntime {

    namespace {
        bool IsConstantInitializer(const GraphViewer &graph, const std::string &initializer_name, bool check_outer_scope) {
            const ONNX_NAMESPACE::TensorProto *initializer = graph.GetConstantInitializer(initializer_name, check_outer_scope);
            return initializer != nullptr;
        }

        bool NodeArgIsConstant(const GraphViewer &graph, const NodeArg &node_arg) {
            return IsConstantInitializer(graph, node_arg.Name(), true);
        }

        namespace actions {
            using NTO = NodesToOptimize;

            class ReplaceGroupNorm : public ReplaceWithNew {
            private:
                std::string OpType(const RuntimeState &) const override { return "GroupNormNhwc"; }

                std::string Domain(const RuntimeState &runtime_state) const override {
                    // TODO we should actually report the node domain that the initial node had
                    // but without instance norm being selected as NHWC this is not possible yet.
                    [[maybe_unused]] const auto& original_domain = runtime_state.selected_nodes.Target().Domain();
                    return kMSInternalNHWCDomain;//
                }

                NodeAttributes ExtraAttributes(const RuntimeState &state) const override {
//                    const auto &in_reshape = *state.selected_nodes.Input(0);
//                    const auto &out_reshape = *state.selected_nodes.Output(0);
// TODO get below numbers from tensors
//                    const ONNX_NAMESPACE::TensorProto *instance_shape = state.graph.GetConstantInitializer(
//                            in_reshape.InputDefs()[1]->Name(), true);
//                    const ONNX_NAMESPACE::TensorProto *shape = state.graph.GetConstantInitializer(
//                            out_reshape.InputDefs()[1]->Name(), true);
                    const int64_t num_groups = 32;//instance_shape->int64_data(1);
                    const auto num_channels = 320;//shape->int64_data(1);
                    ORT_ENFORCE(num_channels % num_groups == 0);
                    NodeAttributes extra_fused_conv_attributes;
//                    Node attribute name for ONNX spec node 18 is different than on crontib op
//                    utils::SetNodeAttribute(utils::MakeAttribute("num_groups", groups), extra_fused_conv_attributes);
                    utils::SetNodeAttribute(utils::MakeAttribute("groups", num_groups), extra_fused_conv_attributes);
                    utils::SetNodeAttribute(utils::MakeAttribute("activation", int64_t(0)), extra_fused_conv_attributes);
                    return extra_fused_conv_attributes;
                }

                std::vector<NodeAndMoveInfo> ValueMoves(const RuntimeState &state) const override {
                    [[maybe_unused]] const auto &instance_norm = state.selected_nodes.Target();
                    [[maybe_unused]] const auto &reshape_in = *state.selected_nodes.Input(0);
                    [[maybe_unused]] const auto &scale = *state.selected_nodes.Output(state.selected_nodes.num_outputs - 1);
                    [[maybe_unused]] const auto &bias = *state.selected_nodes.Output(state.selected_nodes.num_outputs  - 2);

                    const auto reshape_in_location = NTO::NodeLocation{NTO::NodeType::kInput, 0};
                    const auto scale_value_location = NTO::NodeLocation{NTO::NodeType::kOutput, state.selected_nodes.num_outputs - 2};
                    const auto bias_value_location = NTO::NodeLocation{NTO::NodeType::kOutput, state.selected_nodes.num_outputs  - 1};
                    const auto bias_out_location = NTO::NodeLocation{NTO::NodeType::kOutput, state.selected_nodes.num_outputs  - 1};
                    // TODO how do we ignore the multiple dims of bias and scale ?
                    return {
                            MoveAll(reshape_in_location, ArgType::kInput),                              // move output from last reshape
                            MoveToSlot(scale_value_location, ArgType::kInput, 1, ArgType::kInput, 1),   // scale as first input
                            MoveAndAppend(bias_value_location, ArgType::kInput, 1, ArgType::kInput),    // bias as second input
                            MoveAll(bias_out_location, ArgType::kOutput),                               // move output from last reshape
                    };
                }
            };
        } // namespace actions

        namespace selectors {
            class GroupNormSelector : public NodeSelector {
            public:
                GroupNormSelector() = default;

                std::optional<NodesToOptimizeIndices> Select(const GraphViewer &graph_viewer, const Node &node) const override {
                    const auto &node_ep = node.GetExecutionProviderType();
                    // only for CUDA EP
                    if (node_ep != kCudaExecutionProvider) {
                        return std::nullopt;
                    }
//                    Fuse Group Normalization subgraph into one node GroupNorm.
//                    +----------------Shape-------------------------------+
//                    |                                                    |
//                    | (0, 32, -1)                          (B, C, H, W)  v
//     [Root] --> Reshape -------> InstanceNormalization --> Reshape ---> Mul --> Add --> Mul-->  [output]
//    BxCxHxW                 (scale=ones(32), B=zeros(32))               (Cx1x1) (Cx1x1)         Bx512xHxW
//
                    if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "InstanceNormalization", {6}) ||
                        node.GetOutputEdgesCount() != 1) {
                        return std::nullopt;
                    }

                    const auto &in_reshape = *node.InputNodesBegin();
                    const auto &out_reshape = *node.OutputNodesBegin();
                    const auto &scale = *out_reshape.OutputNodesBegin();
                    const auto &bias = *scale.OutputNodesBegin();
                    if (!graph_utils::IsSupportedOptypeVersionAndDomain(in_reshape, "Reshape", {13, 14, 18, 19}) ||
                        !graph_utils::IsSupportedOptypeVersionAndDomain(out_reshape, "Reshape", {13, 14, 18, 19}) ||
                        !graph_utils::IsSupportedOptypeVersionAndDomain(scale, "Mul", {14}) ||
                        !graph_utils::IsSupportedOptypeVersionAndDomain(bias, "Add", {14})) {
                        return std::nullopt;
                    }
                    const std::array<const onnxruntime::Node*, 5> subgraph{&in_reshape, &node, &out_reshape, &scale, &bias};
                    for (const auto current_node: subgraph) {
                        // check that the intermediate outputs are not needed
                        // and all nodes are on the same EP
                        if (current_node->GetInputEdgesCount() != 1 ||
                            current_node->GetOutputEdgesCount() != 1 ||
                            graph_viewer.NodeProducesGraphOutput(*current_node) ||
                            current_node->GetExecutionProviderType() != node.GetExecutionProviderType()) {
                            return std::nullopt;
                        }
                    }

                    // all inputs that are not the tensor data should be constants
                    if (!NodeArgIsConstant(graph_viewer, *in_reshape.InputDefs()[1]) ||
                        !NodeArgIsConstant(graph_viewer, *out_reshape.InputDefs()[1]) ||
                        !NodeArgIsConstant(graph_viewer, *scale.InputDefs()[1]) ||
                        !NodeArgIsConstant(graph_viewer, *bias.InputDefs()[1])) {
                        return std::nullopt;
                    }
                    const ONNX_NAMESPACE::TensorProto *instance_shape = graph_viewer.GetConstantInitializer(
                            in_reshape.InputDefs()[1]->Name(), true);
                    const ONNX_NAMESPACE::TensorProto *shape = graph_viewer.GetConstantInitializer(
                            in_reshape.InputDefs()[1]->Name(), true);
                    const ONNX_NAMESPACE::TensorProto *scale_initializer = graph_viewer.GetConstantInitializer(
                            node.InputDefs()[1]->Name(), true);
                    const ONNX_NAMESPACE::TensorProto *bias_initializer = graph_viewer.GetConstantInitializer(
                            node.InputDefs()[2]->Name(), true);
                    if (scale_initializer == nullptr || bias_initializer == nullptr ||
                        instance_shape == nullptr || shape == nullptr) {
                        return std::nullopt;
                    }
//                    const int num_groups = instance_shape->int64_data(1);
//                    const int num_channels = shape->int64_data(1);
//                    if (bias_initializer->float_data_size() != scale_initializer->float_data_size() ||
//                        scale_initializer->float_data_size() != num_groups ||
//                        bias_initializer->float_data_size() != num_groups) {
//                        return std::nullopt;
//                    }
//                    const auto& scale_values = scale_initializer->float_data();
//                    const auto& bias_values = bias_initializer->float_data();
//                    for (int val_index = 0; val_index < num_groups; ++val_index) {
//                        if (scale_values[val_index] != 1.f ||
//                            bias_values[val_index] != 0.f) {
//                            return std::nullopt;
//                        }
//                    }

                    NodesToOptimizeIndicesBuilder builder{};
                    builder.target_node = node.Index();
                    builder.input_nodes.push_back(in_reshape.Index());
                    builder.output_nodes.push_back(out_reshape.Index());
                    builder.output_nodes.push_back(scale.Index());
                    builder.output_nodes.push_back(bias.Index());
                    return builder.Build();
                };
            };

        }  // namespace selectors

        void RegisterGroupNormDetection(SelectorActionRegistry &registry) {
            const auto name = "GroupNormDetection";
            auto action = std::make_unique<actions::ReplaceGroupNorm>();
#if !defined(ORT_MINIMAL_BUILD)
            auto selector = std::make_unique<selectors::GroupNormSelector>();
            registry.RegisterSelectorAndAction(name, {{"InstanceNormalization", {6}}},
                                               std::move(selector), std::move(action));
#else
            registry.RegisterAction(name, std::move(action));
#endif
        }

        SelectorActionRegistry CreateSelectorActionRegistry() {
            SelectorActionRegistry registry{};
#ifdef ENABLE_CUDA_NHWC_OPS
            RegisterGroupNormDetection(registry);
#endif
            return registry;
        }
    }

    GroupNormDetection::GroupNormDetection(const InlinedHashSet<std::string_view> &compatible_execution_providers,
                                           const SatApplyContextVariant &apply_context)
            : SelectorActionTransformer{
            "GroupNormDetection", CreateSelectorActionRegistry(), apply_context, compatible_execution_providers} {

    }

} // namespace onnxruntime