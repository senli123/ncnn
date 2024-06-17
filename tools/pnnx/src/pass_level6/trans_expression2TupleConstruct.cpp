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

void trans_expression2TupleConstruct(std::shared_ptr<pnnx::Graph> graph)
{
    while (1)
    {
        bool matched = false;

        for (size_t i = 0; i < graph->ops.size(); i++)
        {
            Operator* op = graph->ops[i];
            
            if (op->type != "pnnx.Expression")
                continue;
            // get expr
            if (op->has_param("expr"))
            {
                Parameter param = op->params["expr"];
                std::string expr = param.s;
                // printf("op_name:%s\n",op->name.c_str());
                if (expr.front() == '[' && expr.back() == ']')
                {
                    matched = true;
                    std::vector<Operand*> outputs = op->outputs;
                    bool sink_node_is_index = false;
                    if(outputs[0]->consumers[0]->type == "Tensor.index")
                    {
                        sink_node_is_index = true;
                    }
                   
                    if (sink_node_is_index)
                    {
                        // update expr 
                        std::string out_operand_name = outputs[0]->name;
                        size_t pos = 0; 
                        if((pos = expr.find("0")) != std::string::npos)
                        {
                            expr.replace(pos, 1, out_operand_name);
                        }
                        outputs[0]->consumers[0]->params["expr"] = expr;
                        Operand* input = op->inputs[0];
                        Operator* pre_node = input->producer;  
                        pre_node->outputs.clear();
                        for (auto& single_out : outputs)
                        {
                            single_out->producer = pre_node;
                            pre_node->outputs.push_back(single_out);
                        }
                        input->producer = 0;
                        input->consumers.clear();
                        graph->operands.erase(std::find(graph->operands.begin(), graph->operands.end(), input));
                        delete input;

                        op->inputs.clear();
                        op->outputs.clear();

                        graph->ops.erase(graph->ops.begin() + i);
                        delete op;
                    }
                    else
                    {
                        op->type = "prim::TupleConstruct";
                        op->params.clear();
                    }
                   
                    break;
                }
            }
        }

        if (!matched)
            break;
    }
}

} // namespace pnnx
