// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "eliminate_ListUnpack.h"

#include <algorithm>
#include "pass_level2.h"

namespace pnnx {

void eliminate_ListUnpack(Graph& graph)
{
    while (1)
    {
        bool matched = false;

        for (size_t i = 0; i < graph.ops.size(); i++)
        {
            Operator* op = graph.ops[i];

            if (op->type != "prim::ListUnpack")
                continue;

            // delete noop-like prim::ListUnpack
            matched = true;
            Operand* ListUnpack_input = op->inputs[0];             //  get cur node input
            std::vector<Operand*> ListUnpack_output = op->outputs; // get cur node output
            Operator* pre_node = ListUnpack_input->producer;       //get pre node
            pre_node->outputs.clear();
            for (auto& single_out : ListUnpack_output)
            {
                single_out->producer = pre_node;
                pre_node->outputs.push_back(single_out);
            }
            ListUnpack_input->producer = 0;
            ListUnpack_input->consumers.clear();
            graph.operands.erase(std::find(graph.operands.begin(), graph.operands.end(), ListUnpack_input));
            delete ListUnpack_input;

            op->inputs.clear();
            op->outputs.clear();

            graph.ops.erase(graph.ops.begin() + i);
            delete op;

            break;
        }

        if (!matched)
            break;
    }
}

} // namespace pnnx
