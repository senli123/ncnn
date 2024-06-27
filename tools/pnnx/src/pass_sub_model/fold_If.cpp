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

#include "fold_If.h"

#include <algorithm>
#include "pass_level2.h"

namespace pnnx {

void fold_If(std::shared_ptr<pnnx::MainGraph> graph)
{
    std::queue<std::shared_ptr<pnnx::MainGraph>> main_graph_queue;  
    main_graph_queue.push(graph);
    while( !main_graph_queue.empty())
    {
        std::shared_ptr<pnnx::MainGraph> cur_graph = main_graph_queue.front();
        main_graph_queue.pop();
       
        std::shared_ptr<pnnx::Graph> main_graph = cur_graph->get_main_graph();       
        for(auto pair : cur_graph->op_2_graph)
        {
            std::string name = pair.first;
            std::unordered_map<std::string, std::vector<int>> block_info = pair.second;
            Operator* op = main_graph->get_operator(name);
            if(op->type == "prim::If")
            {
                op->type = "pnnx.If";
                // fold condition, iter_num tensor index to params
                Operand* if_condition_input = op->inputs[0];
                
                Operator* if_condition_op = if_condition_input->producer;
                if(if_condition_op->type == "pnnx.Expression")
                {
                   
                    // fold sub model to base model
                    std::string if_condition_data = if_condition_op->params["expr"].s;
                    if(if_condition_data == "False")
                    {
                        // get sub model
                        std::string sub_if_op_block_name = op->name + "_block1";
                        std::shared_ptr<pnnx::MainGraph> sub_if_model = cur_graph->get_sub_graph(sub_if_op_block_name);
                        // connect sub model to main model
                        std::shared_ptr<pnnx::Graph> sub_if_main_model = sub_if_model->get_main_graph();
                        // process input
                        int input_size = op->inputs.size();
                        for(int i = 1; i < input_size; i++)
                        {
                            // get  base model input operand
                            Operand* base_model_operand = op->inputs[i];
                            base_model_operand->consumers.erase(std::find(base_model_operand->consumers.begin(), base_model_operand->consumers.end(), op));
                            // get sub model input
                            std::string input_op_name = "pnnx_input_" + std::to_string(i-1);
                            Operator* input_op = sub_if_main_model->get_operator(input_op_name);
                            // consumers op
                            Operand* input_operand =  input_op->outputs[0];
                            std::vector<Operator*> consumers = input_operand->consumers;
                            for(auto single_consumer: consumers)
                            {
                                base_model_operand->consumers.push_back(single_consumer);
                                single_consumer->inputs.erase(std::find(single_consumer->inputs.begin(),single_consumer->inputs.end(), input_operand));
                                single_consumer->inputs.push_back(base_model_operand);
                                
                            }
                            // delete input_operand
                            input_operand->producer = 0;
                            input_operand->consumers.clear();
                            sub_if_main_model->operands.erase(std::find(sub_if_main_model->operands.begin(), sub_if_main_model->operands.end(), input_operand));
                            delete input_operand;
                            // delete input op
                            input_op->inputs.clear();
                            input_op->outputs.clear();
                            sub_if_main_model->ops.erase(std::find(sub_if_main_model->ops.begin(), sub_if_main_model->ops.end(), input_op));
                            delete input_op;

                        }

                        // process output
                        int output_size = op->outputs.size();
                        for(int i = 0; i < output_size; i++)
                        {
                            // get  base model output operand
                            Operand* base_model_output_operand = op->outputs[i];
                            // get sub model output
                            std::string output_op_name = "pnnx_output_" + std::to_string(i);
                            Operator* output_op = sub_if_main_model->get_operator(output_op_name);
                            // producer op
                            Operand* output_operand =  output_op->inputs[0];
                            Operator* producer = output_operand->producer;
                            base_model_output_operand->producer = producer;

                            producer->outputs.erase(std::find( producer->outputs.begin(), producer->outputs.end(), output_operand));
                            producer->outputs.push_back(base_model_output_operand);
                            // delete output_operand
                            output_operand->producer = 0;
                            output_operand->consumers.clear();
                            sub_if_main_model->operands.erase(std::find(sub_if_main_model->operands.begin(), sub_if_main_model->operands.end(), output_operand));
                            delete output_operand;
                            // delete output op
                            output_op->inputs.clear();
                            output_op->outputs.clear();
                            sub_if_main_model->ops.erase(std::find(sub_if_main_model->ops.begin(), sub_if_main_model->ops.end(), output_op));
                            delete output_op;
                            
                        }



                        // copy sub model ops operand to base model
                        main_graph->ops.insert(main_graph->ops.end(), sub_if_main_model->ops.begin(), sub_if_main_model->ops.end()); 
                        main_graph->operands.insert(main_graph->operands.end(), sub_if_main_model->operands.begin(), sub_if_main_model->operands.end());
                        //earse sub model
                        // cur_graph->op_2_graph.erase(op->name);
                        // std::string sub_if_op_block_name0 = op->name + "_block0";
                        // std::string sub_if_op_block_name1 = op->name + "_block1";
                        // cur_graph->sub_graph_map.erase(sub_if_op_block_name0);
                        // cur_graph->sub_graph_map.erase(sub_if_op_block_name1);

                        // earse sub_model in effective_sub_model_name
                        std::string sub_if_op_block_name0 = op->name + "_block0";
                        std::string sub_if_op_block_name1 = op->name + "_block1";
                        cur_graph->effective_sub_model_name.erase(std::find(cur_graph->effective_sub_model_name.begin(), cur_graph->effective_sub_model_name.end(), sub_if_op_block_name0));
                        cur_graph->effective_sub_model_name.erase(std::find(cur_graph->effective_sub_model_name.begin(), cur_graph->effective_sub_model_name.end(), sub_if_op_block_name1));
                        
                        // delete if_condition_input
                        if_condition_input->producer = 0;
                        if_condition_input->consumers.clear();
                        main_graph->operands.erase(std::find(main_graph->operands.begin(), main_graph->operands.end(), if_condition_input));
                        delete if_condition_input;
                        
                        // delete if_condition_op
                        if_condition_op->inputs.clear();
                        if_condition_op->outputs.clear();
                        main_graph->ops.erase(std::find(main_graph->ops.begin(), main_graph->ops.end(), if_condition_op));
                        delete if_condition_op;

                        //delete op
                        op->inputs.clear();
                        op->outputs.clear();
                        main_graph->ops.erase(std::find(main_graph->ops.begin(), main_graph->ops.end(), op));
                        delete op;  
                    }
                }
                else
                {
                    // fold info 
                    std::vector<std::string> block_names;
                    for(auto block_info_pair : block_info)
                    {
                        block_names.push_back(block_info_pair.first);
                        std::vector<int> input_indexes = block_info_pair.second;
                        op->params[block_info_pair.first + "_input_indexes"] = input_indexes;

                    }
                    op->params["block_names"] = block_names; 
                }
                
            }
 
        }
        
        for(auto pair: cur_graph->sub_graph_map)
        {
            auto it = std::find(cur_graph->effective_sub_model_name.begin(), cur_graph->effective_sub_model_name.end(), pair.first);
            if(it!= cur_graph->effective_sub_model_name.end())
            {
                main_graph_queue.push(pair.second);
            }
            
        }  
    }
}

} // namespace pnnx
