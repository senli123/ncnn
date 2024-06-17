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

#include "fold_Loop.h"

#include <algorithm>
#include "pass_level2.h"

namespace pnnx {

void fold_Loop(std::shared_ptr<pnnx::Graph> graph)
{
    while (1)
    {
        bool matched = false;

        for (size_t i = 0; i < graph->ops.size(); i++)
        {
            Operator* op = graph->ops[i];

            if (op->type != "prim::Loop")
                continue;
            op->type = "pnnx.Loop";
            // delete prim::Loop
            matched = true;
            Operand* loop_iterNum_input = op->inputs[0];
            Operand* loop_condition_input = op->inputs[1];

            op->inputs.erase(op->inputs.begin());  
            op->inputs.erase(op->inputs.begin()); 
            // parse iterNum only used for static
            // [todo] dynamic
            Operator* loop_iterNum_expression = loop_iterNum_input->producer;
            std::string iterNum_expr = loop_iterNum_expression->params["expr"].s;
            // check pre_node or not
            if(loop_iterNum_expression->inputs.size() == 0)
            {
                int iter_num = std::stoi(iterNum_expr);
                op->params["iter_num"] = iter_num;
            }
            else{
                Operand* pre_loop_iterNum_input = loop_iterNum_expression->inputs[0];
                op->params["iter_num"] = pre_loop_iterNum_input->shape[0];
                pre_loop_iterNum_input->consumers.erase(std::find(pre_loop_iterNum_input->consumers.begin(),  pre_loop_iterNum_input->consumers.end(), loop_iterNum_expression));
                loop_iterNum_expression->inputs.clear();
            }
            // delete iterNum expression
            loop_iterNum_input->producer = 0;
            loop_iterNum_input->consumers.clear();
            graph->operands.erase(std::find(graph->operands.begin(), graph->operands.end(), loop_iterNum_input));
            delete loop_iterNum_input;

            loop_iterNum_expression->inputs.clear();
            loop_iterNum_expression->outputs.clear();
            graph->ops.erase(std::find(graph->ops.begin(), graph->ops.end(), loop_iterNum_expression));
            delete loop_iterNum_expression;

            // parse condition
            Operator* loop_condition_expression = loop_condition_input->producer;
            std::string condition_expr = loop_condition_expression->params["expr"].s;
            op->params["condition"] = condition_expr;

            // delete condition expression
            loop_condition_input->producer = 0;
            loop_condition_input->consumers.clear();
            graph->operands.erase(std::find(graph->operands.begin(), graph->operands.end(), loop_condition_input));
            delete loop_condition_input;

            loop_condition_expression->inputs.clear();
            loop_condition_expression->outputs.clear();
            graph->ops.erase(std::find(graph->ops.begin(), graph->ops.end(), loop_condition_expression));
            delete loop_condition_expression;

            break;
        }

        if (!matched)
            break;
    }
}

} // namespace pnnx
