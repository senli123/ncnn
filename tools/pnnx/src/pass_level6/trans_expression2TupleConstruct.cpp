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

#include "trans_expression2TupleConstruct.h"

#include <algorithm>
#include "pass_level2.h"

namespace pnnx {

void trans_expression2TupleConstruct(Graph& graph)
{
    while (1)
    {
        bool matched = false;

        for (size_t i = 0; i < graph.ops.size(); i++)
        {
            Operator* op = graph.ops[i];

            if (op->type != "pnnx.Expression")
                continue;
            // get expr
            if (op->has_param("expr"))
            {
                Parameter param = op->params["expr"];
                std::string expr = param.s;
                if (expr.front() == '[' && expr.back() == ']')
                {
                    matched = true;
                    op->type = "prim::TupleConstruct";
                    op->params.clear();
                    break;
                }
            }
        }

        if (!matched)
            break;
    }
}

} // namespace pnnx
