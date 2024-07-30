// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "fold_constants_sub_graph.h"
#include <unordered_set>

#include "storezip.h"
#include "pass_level4/dead_code_elimination.h"

namespace pnnx {

void fold_constants_sub_graph(std::shared_ptr<pnnx::Graph> graph)
{

    // StoreZipReader zip;
    // zip.open(all_tensor_zippath);

    // find input node [like pnnx.Input torch.arrage]
    // Attribute node like[pnnx.Attribute]
    std::queue<Operator*> input_node_list;
    std::queue<Operator*> attribute_node_list;
    for(size_t i = 0; i < graph->ops.size(); i++)
    {
        Operator* cur_op = graph->ops[i];
        if (cur_op->type == "pnnx.Input")
        {
            input_node_list.push(cur_op);
        }
        else if(cur_op->type == "pnnx.Attribute" || cur_op->inputs.size() == 0)
        {
            
            attribute_node_list.push(cur_op);
        }
    }

    // Assign the nodes associated with the input nodes as main labels
    std::queue<Operator*> main_node_list;
    while(!input_node_list.empty())
    {
        Operator* input_node = input_node_list.front();
        input_node_list.pop();
        input_node->label = "main";
        std::vector<Operand*> output_operands = input_node->outputs;
        for(auto out: output_operands)
        {
            auto consumers = out->consumers;
            for(auto consumer: consumers)
            {
                if(consumer->label == "")
                {
                    consumer->label = "main";
                    main_node_list.push(consumer);
                }
            }
        }
        while(!main_node_list.empty())
        {
            Operator* cur_node = main_node_list.front();
            main_node_list.pop();
            std::vector<Operand*> output_operands = cur_node->outputs;
            for(auto out: output_operands)
            {
                auto consumers = out->consumers;
                for(auto consumer: consumers)
                {
                    if(consumer->label == "")
                    {
                        consumer->label = "main";
                        main_node_list.push(consumer);
                    }
                }
            }
        }
    }
    // Assign the nodes associated with the attribute nodes as main, delete or replace labels
    std::queue<Operator*> delete_replace_node_list;
    while(!attribute_node_list.empty())
    {
        Operator* input_attribute_node = attribute_node_list.front();
        attribute_node_list.pop();
        std::vector<Operand*> output_operands = input_attribute_node->outputs;
        for(auto out: output_operands)
        {
            auto consumers = out->consumers;
            for(auto consumer: consumers)
            {
                if(consumer->label == "")
                {
                    delete_replace_node_list.push(consumer);
                }
                else if(consumer->label == "main")
                {
                    input_attribute_node->label = "main";
                }
                
            }
        }
        
        if(input_attribute_node->label == "")
        {
            input_attribute_node->label = "delete";
        }
        
        while(!delete_replace_node_list.empty())
        {
            Operator* cur_node = delete_replace_node_list.front();
            delete_replace_node_list.pop();
           
            std::vector<Operand*> output_operands = cur_node->outputs;
            bool consumer_node_is_main = false;
            for(auto out: output_operands)
            {
                auto consumers = out->consumers;
                for(auto consumer: consumers)
                {
                   if(consumer->label == "")
                    {
                        delete_replace_node_list.push(consumer);
                    }
                    else if(consumer->label == "main")
                    {
                        consumer_node_is_main = true;
                    }
                }
            }
            if(consumer_node_is_main)
            {
                cur_node->label = "replace";
            }
            else
            {
                cur_node->label = "delete";
            }
        }

    }
    
    // process replace node an delete node
    // while(true)
    // {
    //     bool matched = false;
    //     for(size_t i = 0; i < graph->ops.size(); i++)
    //     {
    //         Operator* cur_op = graph->ops[i];
    //         if (cur_op->label == "")
    //         {
    //             fprintf(stderr, "############# find a not sign node, node name is: %s\n",  cur_op->name.c_str());
    //         }
    //         else if(cur_op->label == "replace")
    //         {
    //             matched = true;
    //             std::vector<Operand*> outputs = cur_op->outputs;
    //             for(auto out: outputs)
    //             {
    //                 std::string name = out->name;
    //                 // replace cur op to pnnx.Attribute node
    //                 Operator* op_new = graph->new_operator_after("pnnx.Attribute", std::string("pnnx_fold_") + name, cur_op);
    //                 op_new->label == "main";
    //                 op_new->attrs["data"] = Attribute();
    //                 Attribute& t2 = op_new->attrs["data"];
    //                 t2.type = out->type;
    //                 t2.shape = out->shape;
    //                 size_t size = zip.get_file_size(name);
    //                 t2.data.resize(size);
    //                 zip.read_file(name, t2.data.data());

    //                 op_new->outputs.push_back(out);
    //                 out->producer = op_new;       
    //             }
    //             std::vector<Operand*> cur_op_inputs = cur_op->inputs;
    //             for(auto input: cur_op_inputs)
    //             {
    //                 input->consumers.erase(std::find(input->consumers.begin(), input->consumers.end(), cur_op));
    //             }
    //             cur_op->inputs.clear();
    //             cur_op->outputs.clear();
    //             graph->ops.erase(graph->ops.begin() + i);
    //             delete cur_op;
    //             break;
    //         }
    //         else if(cur_op->label == "delete")
    //         {
    //             matched = true;
    //             std::vector<Operand*> cur_op_inputs = cur_op->inputs;
    //             for(auto input: cur_op_inputs)
    //             {
    //                 input->consumers.erase(std::find(input->consumers.begin(), input->consumers.end(), cur_op));
    //             }
    //             std::vector<Operand*> outputs = cur_op->outputs;
    //             for(auto out: outputs)
    //             {
    //                 for(auto consumer: out->consumers)
    //                 {
    //                     consumer->inputs.erase(std::find(consumer->inputs.begin(), consumer->inputs.end(), out));
    //                 }
    //             } 
    //             for(auto out: outputs)
    //             {
    //                 out->producer = 0;
    //                 out->consumers.clear();
    //                 graph->operands.erase(std::find(graph->operands.begin(), graph->operands.end(), out));
    //                 delete out;
    //             } 
    //             cur_op->inputs.clear();
    //             cur_op->outputs.clear();

    //             graph->ops.erase(graph->ops.begin() + i);
    //             delete cur_op;
    //             break;
    //         }
    //     }
    //     if (!matched)
    //         break;

    // }

    // zip.close();
    // dce
    dead_code_elimination(graph);
}

} // namespace pnnx
