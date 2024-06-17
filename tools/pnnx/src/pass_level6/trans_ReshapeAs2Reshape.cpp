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

#include "trans_ReshapeAs2Reshape.h"

#include <algorithm>
#include "pass_level2.h"

namespace pnnx {

void trans_ReshapeAs2Reshape(std::shared_ptr<pnnx::Graph> graph)
{
    while (1)
    {
        bool matched = false;

        for (size_t i = 0; i < graph->ops.size(); i++)
        {
            Operator* op = graph->ops[i];

            if (op->type != "Tensor.reshape_as")
                continue;
            matched = true;
            // get the input size of input1
            Operand* input1 = op->inputs.at(1);
            std::vector<int> input_shape = input1->shape;
            // reshape_as 2 reshape
            op->type = "Tensor.reshape";
            op->params["shape"] = input_shape;
            op->inputs.pop_back();  

            // if input1 is only used for reshape_as, delete its op and operand in graph
            // if (input1->consumers.size() == 1)
            // {

            //     Operator* pre_producer = input1->producer;
            //     for(auto out : pre_producer->outputs)
            //     {
            //         if (out->name == input1->name)
            //         {
            //             std::swap(out, pre_producer->outputs.back());  
            //             pre_producer->outputs.pop_back();
            //             break; 
            //         }
            //     }

            //     for(int index = 0; index < graph->operands.size(); index++)
            //     {
            //         if(graph->operands[index]->name == input1->name)
            //         {
            //             graph->operands.erase(graph->operands.begin() + index);
            //             break;
            //         }
            //     }
            // }
            std::vector<Operand*> delete_operands = {};
            std::vector<Operator*> delete_ops = {};
            std::vector<Operand*>  operand_squence = {input1};
            while(operand_squence.size() > 0)
            {
                Operand* cur_operand = operand_squence.front();
                operand_squence.erase(operand_squence.begin());
                if (cur_operand->consumers.size() == 1)
                {
                    delete_operands.push_back(cur_operand);
                    Operator* pre_producer = cur_operand->producer;
                    if(pre_producer->outputs.size() == 1)
                    {
                        delete_ops.push_back(pre_producer);
                        for(auto cur_input: pre_producer->inputs)
                        {
                            operand_squence.push_back(cur_input);
                        }
                    }
                    else
                    {
                        for(auto out : pre_producer->outputs)
                        {
                            if (out->name == input1->name)
                            {
                                std::swap(out, pre_producer->outputs.back());  
                                pre_producer->outputs.pop_back();
                                break; 
                            }
                        }
                    }
                }
                else
                {
                    for(int index = 0; index < cur_operand->consumers.size(); index++)
                    {
                        if(cur_operand->consumers[index]->name == op->name)
                        {
                            cur_operand->consumers.erase(cur_operand->consumers.begin() + index);
                        }
                    }
                }
            }

            for(auto delete_op: delete_ops)
            {
                delete_op->inputs.clear();
                delete_op->outputs.clear();
                graph->ops.erase(std::find(graph->ops.begin(), graph->ops.end(), delete_op));
                delete delete_op;
            }
            for(auto delete_operand: delete_operands)
            {
                graph->operands.erase(std::find(graph->operands.begin(), graph->operands.end(), delete_operand));
                delete delete_operand;
            }
            break;
           
        }

        if (!matched)
            break;
    }
}

} // namespace pnnx
