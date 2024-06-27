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

void fold_Loop(std::shared_ptr<pnnx::MainGraph> graph)
{
    std::queue<std::shared_ptr<pnnx::MainGraph>> main_graph_queue;  
    main_graph_queue.push(graph);
    while( !main_graph_queue.empty())
    {
        std::shared_ptr<pnnx::MainGraph> cur_main_graph = main_graph_queue.front();
        main_graph_queue.pop();
        std::shared_ptr<pnnx::Graph> graph = cur_main_graph->get_main_graph(); 
        for(auto pair: cur_main_graph->sub_graph_map)
        {
            main_graph_queue.push(pair.second);
        }        
        for(auto pair : cur_main_graph->op_2_graph)
        {
            std::string name = pair.first;
            std::unordered_map<std::string, std::vector<int>> block_info = pair.second;
            Operator* op = graph->get_operator(name);
            if(op->type == "prim::Loop")
            {
                op->type = "pnnx.Loop";
                // fold condition, iter_num tensor index to params
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

                std::vector<std::string> block_names;
                for(auto block_info_pair : block_info)
                {
                    block_names.push_back(block_info_pair.first);
                    std::vector<int> input_indexes = block_info_pair.second;
                    for (auto& num : input_indexes)
                    {  
                        num -= 2;  
                    }
                    input_indexes.erase(input_indexes.begin());
                    input_indexes.erase(input_indexes.begin());
                    op->params[block_info_pair.first + "_input_indexes"] = input_indexes;

                }
                op->params["block_names"] = block_names;
            }

        }
    }
}

} // namespace pnnx
