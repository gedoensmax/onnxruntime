// Copyright (c) NVIDIA Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/selectors_actions/selector_action_transformer.h"

namespace onnxruntime {

/**
@Class GroupNormDetection

Selector that detects combinations of Reshape+Instance Norm that could be replaced by a GroupNorm

It is attempted to be triggered only on nodes with op type "Reshape".
*/
    class GroupNormDetection : public SelectorActionTransformer {
    public:
        GroupNormDetection(const InlinedHashSet<std::string_view>& compatible_execution_providers = {},
                           const SatApplyContextVariant& apply_context = {});
    };

} // onnxruntime

