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
                std::string OpType(const RuntimeState &) const override { return "GroupNorm"; }

                std::string Domain(const RuntimeState &runtime_state) const override {
                    return kMSDomain;
                }

                NodeAttributes ExtraAttributes(const RuntimeState &state) const override {
                    const auto &first_node = *state.selected_nodes.Input(0);
                    bool nhwc = first_node.OpType() == "Transpose";

                    const auto &in_reshape = *state.selected_nodes.Input(nhwc);
                    const auto &out_reshape = *state.selected_nodes.Output(state.selected_nodes.num_outputs - (3 + nhwc));
                    const ONNX_NAMESPACE::TensorProto *instance_shape, *shape;
                    ORT_ENFORCE(state.graph.GetInitializedTensor(
                            in_reshape.InputDefs()[1]->Name(), instance_shape));
                    ORT_ENFORCE(state.graph.GetInitializedTensor(
                            out_reshape.InputDefs()[1]->Name(), shape));

                    // shape on which the instance norm is executed
                    Initializer instance_shape_init(*instance_shape, state.graph.ModelPath());
                    // the actual data shape to which we go back after the instance norm
//                    Initializer shape_init(*shape, state.graph.ModelPath());
                    const int64_t num_groups = instance_shape_init.data<int64_t>()[1];
//                    const int64_t num_channels = shape_init.data<int64_t>()[1];
                    NodeAttributes extra_fused_conv_attributes;
                    utils::SetNodeAttribute(utils::MakeAttribute("groups", num_groups), extra_fused_conv_attributes);
                    utils::SetNodeAttribute(utils::MakeAttribute("activation", int64_t(0)), extra_fused_conv_attributes);
                    utils::SetNodeAttribute(utils::MakeAttribute("channels_last", int64_t(nhwc ? 1 : 0)), extra_fused_conv_attributes);
                    return extra_fused_conv_attributes;
                }

                std::vector<NodeAndMoveInfo> ValueMoves(const RuntimeState &state) const override {
                    const auto &first_node = *state.selected_nodes.Input(0);
                    bool nhwc = first_node.OpType() == "Transpose";

                    const auto reshape_in_location = NTO::NodeLocation{NTO::NodeType::kInput, (0 + nhwc)};
                    const auto scale_value_location = NTO::NodeLocation{NTO::NodeType::kOutput, state.selected_nodes.num_outputs - (2 + nhwc)};
                    const auto bias_value_location = NTO::NodeLocation{NTO::NodeType::kOutput, state.selected_nodes.num_outputs - (1 + nhwc)};
                    const auto bias_out_location = NTO::NodeLocation{NTO::NodeType::kOutput, state.selected_nodes.num_outputs - (1 + nhwc)};
                    // TODO how do we ignore the multiple dims of bias and scale tensors / squeeze them
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
                    const std::array<const onnxruntime::Node *, 5> subgraph{&in_reshape, &node, &out_reshape, &scale, &bias};
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

                    const ONNX_NAMESPACE::TensorProto *instance_shape, *shape, *scale_tensor, *bias_tensor;
                    if (!graph_viewer.GetInitializedTensor(
                            in_reshape.InputDefs()[1]->Name(), instance_shape)) {
                        return std::nullopt;
                    }
                    if (!graph_viewer.GetInitializedTensor(
                            out_reshape.InputDefs()[1]->Name(), shape)) {
                        return std::nullopt;
                    }

                    Initializer instance_shape_init(*instance_shape, graph_viewer.ModelPath());
                    Initializer shape_init(*shape, graph_viewer.ModelPath());
                    const int64_t num_groups = instance_shape_init.data<int64_t>()[1];
                    const int64_t num_channels = shape_init.data<int64_t>()[1];
                    if (num_channels % num_groups != 0) {
                        return std::nullopt;
                    }

                    // check if instance norm has all 1 scale and all 0 bias
                    if (!graph_viewer.GetInitializedTensor(
                            node.InputDefs()[1]->Name(), scale_tensor)) {
                        return std::nullopt;
                    }
                    if (!graph_viewer.GetInitializedTensor(
                            node.InputDefs()[2]->Name(), bias_tensor)) {
                        return std::nullopt;
                    }

                    Initializer scale_init(*scale_tensor, graph_viewer.ModelPath());
                    Initializer bias_init(*bias_tensor, graph_viewer.ModelPath());
                    if (scale_init.size() != bias_init.size() ||
                        scale_init.size() != size_t(num_groups) ||
                        bias_init.size() != size_t(num_groups)) {
                        return std::nullopt;
                    }
                    [[maybe_unused]] auto dtype = scale_init.data_type();
                    // TODO respect data type fp32/fp16 here
                    const auto *scale_values = scale_init.data<float>();
                    const auto *bias_values = bias_init.data<float>();
                    for (int val_index = 0; val_index < num_groups; ++val_index) {
                        if (scale_values[val_index] != 1.f ||
                            bias_values[val_index] != 0.f) {
                            return std::nullopt;
                        }
                    }
                    // check if node is NHWC
                    // TODO currently the group norm is not wrapped in transforms as the tranpose is "pushed" through the network by trnaspose optimizer
                    const auto &in_transpose = *in_reshape.InputNodesBegin();
                    const auto &out_transpose = *bias.OutputNodesBegin();
                    bool nhwc_group_norm = in_transpose.OpType() == "Transpose" && out_transpose.OpType() == "Transpose";

                    NodesToOptimizeIndicesBuilder builder{};
                    builder.target_node = node.Index();
                    if (nhwc_group_norm) {
                        builder.input_nodes.push_back(in_transpose.Index());
                    }
                    builder.input_nodes.push_back(in_reshape.Index());
                    builder.output_nodes.push_back(out_reshape.Index());
                    builder.output_nodes.push_back(scale.Index());
                    builder.output_nodes.push_back(bias.Index());
                    if (nhwc_group_norm) {
                        builder.output_nodes.push_back(out_transpose.Index());
                    }
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