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
#include <torch/csrc/jit/passes/quantization/helper.h>
#include <torch/csrc/api/include/torch/version.h>

#include "pass_level1.h"

namespace pnnx {

FuseModulePass::~FuseModulePass()
{
}

void FuseModulePass::write(Operator* /*op*/, const std::shared_ptr<torch::jit::Graph>& /*graph*/) const
{
}

void FuseModulePass::write(Operator* op, const std::shared_ptr<torch::jit::Graph>& graph, const torch::jit::Module& /*mod*/) const
{
    write(op, graph);
}

static std::vector<const FuseModulePass*> g_global_pnnx_fuse_module_passes;

const std::vector<const FuseModulePass*>& get_global_pnnx_fuse_module_passes()
{
    return g_global_pnnx_fuse_module_passes;
}

FuseModulePassRegister::FuseModulePassRegister(const FuseModulePass* _pass)
    : pass(_pass)
{
    g_global_pnnx_fuse_module_passes.push_back(pass);
}

FuseModulePassRegister::~FuseModulePassRegister()
{
    delete pass;
}

static void fuse_moduleop_unpack(std::shared_ptr<Graph>& graph, const std::vector<std::string>& module_operators)
{
    while (1)
    {
        bool matched = false;

        for (size_t i = 0; i < graph->ops.size(); i++)
        {
            Operator* op = graph->ops[i];

            if (std::find(module_operators.begin(), module_operators.end(), op->type) == module_operators.end())
                continue;

            if (op->outputs.size() != 1)
                continue;

            if (op->outputs[0]->consumers.size() != 1)
                continue;

            Operator* op2 = op->outputs[0]->consumers[0];
            if (op2->type != "prim::TupleUnpack")
                continue;

            matched = true;

            op->outputs[0]->producer = 0;
            op->outputs[0]->remove_consumer(op2);

            for (auto& x : op2->outputs)
            {
                x->producer = op;
            }

            op->outputs = op2->outputs;

            op2->inputs.clear();
            op2->outputs.clear();

            graph->ops.erase(std::find(graph->ops.begin(), graph->ops.end(), op2));

            delete op2;

            break;
        }

        if (!matched)
            break;
    }
}


void pass_level1_block(const torch::jit::Module& mod,  Operator* src_op, torch::jit::Block* sub_block, std::shared_ptr<Graph>& sub_pnnx_graph, \
const std::vector<std::string>& module_operators, std::unordered_map<std::string, std::shared_ptr<pnnx::Graph>>& pnnx_graph_map, \
int& pnnx_unknown_index, int& pnnx_loop_index)
{
    // create_input
    int last_input_op_index = 0;
    
    for (int i = 1; i < (int)sub_block->inputs().size(); i++)
    {
        char input_name[32];
        sprintf(input_name, "pnnx_input_%d", i - 1);
        last_input_op_index = i - 1;
        const auto& block_input = sub_block->inputs()[i];
        // block_input->debugName()
        Operator* op = sub_pnnx_graph->new_operator("pnnx.Input", input_name);
        Operand* r = sub_pnnx_graph->new_operand(block_input);
        Operand* src_r =  src_op->inputs.at(i + 1);
        r->params = src_r->params;
        r->type = src_r->type;
        r->shape = src_r->shape;
        r->producer = op;
        op->outputs.push_back(r);
    }
    std::map<std::string, std::string> class_type_to_names;
    for (const auto& n : sub_block->nodes())
    {
        if (n->kind() == c10::prim::GetAttr)
        {
            // pass
            std::string name = n->s(torch::jit::attr::name);
            //             std::string name = n->debugName();

            auto class_type = n->output(0)->type()->cast<torch::jit::ClassType>();

            if (class_type)
            {
                std::string class_type_str = class_type->str();
                class_type_to_names[class_type_str] = name;
                //             class_type_to_names[class_type_str] = class_type_str + "." + name;
            }
            else
            {
                // Tensor from some class
                //                 Operator* op = pg->new_operator(n->kind().toDisplayString(), name);
                Operator* op = sub_pnnx_graph->new_operator("pnnx.Attribute", name);

                for (int i = 0; i < (int)n->outputs().size(); i++)
                {
                    const auto& on = n->output(i);
                    Operand* r = sub_pnnx_graph->new_operand(on);
                    r->producer = op;
                    op->outputs.push_back(r);
                }

                std::deque<std::string> module_names; // = split(n->input(0)->node()->s(torch::jit::attr::name), '.');
                {
                    auto np = n->input(0)->node();
                    while (np->hasAttribute(torch::jit::attr::name))
                    {
                        module_names.push_front(np->s(torch::jit::attr::name));
                        np = np->input(0)->node();
                    }
                }

                std::string wrapped_name;
                auto sub_mod = mod;
                for (auto module_name : module_names)
                {
                    if (wrapped_name.size() > 0)
                        wrapped_name = wrapped_name + "." + module_name;
                    else
                        wrapped_name = module_name;
                    sub_mod = sub_mod.attr(module_name).toModule();
                }

                if (wrapped_name.empty())
                {
                    // top-level module
                    wrapped_name = name;
                }

                op->name = wrapped_name;

                // op->params["this"] = n->input(i)

                // sub_mod.dump(true, true, true);

                op->attrs["data"] = sub_mod.attr(name).toTensor();
                op->outputs[0]->type = op->attrs["data"].type;
                op->outputs[0]->shape = op->attrs["data"].shape;
            }
        }
        else if (n->kind() == c10::prim::Constant) // || n->kind() == c10::prim::ListConstruct)
        {
            char name[32];
            sprintf(name, "pnnx_%d", pnnx_unknown_index++);

            Operator* op = sub_pnnx_graph->new_operator(n->kind().toDisplayString(), name);

            for (int i = 0; i < (int)n->inputs().size(); i++)
            {
                const auto& in = n->input(i);
                Operand* r = sub_pnnx_graph->get_operand(in->debugName());
                if (r == 0)
                {
                    Operand* r1  = 0;
                    for (const auto& graph_pair : pnnx_graph_map) {  
                        r1 = graph_pair.second->get_operand(in->debugName());
                        if(r1 != 0)
                        break;
                    }  
                
                    assert(r1 != 0 && "cur tensor name : %s not in graph\n",in->debugName().c_str());

                    // if(r1 == 0)
                    // {
                    //     throw std::exception("cur tensor name : %s not in graph\n",in->debugName().c_str());
                    // }
                    std::string last_input_op_name =  "pnnx_input_" + std::to_string(last_input_op_index);
                    Operator* last_input_op = sub_pnnx_graph->get_operator(last_input_op_name);
                    assert(last_input_op != 0 && "failed to find last input op : %s\n",last_input_op_name.c_str());
                    if(r1->producer->type == "prim::Constant")
                    {
                        Operator* constant_op = r1->producer;
                        // insert type of prim::Constant new input to sub_graph
                        Operator* new_input_op = sub_pnnx_graph->new_operator_after(constant_op->type, constant_op->name, last_input_op);
                        new_input_op->inputnames = constant_op->inputnames;
                        new_input_op->params = constant_op->params;
                        new_input_op->attrs = constant_op->attrs;
                        Operand* r2 = sub_pnnx_graph->new_operand(in->debugName());
                        r2->producer = new_input_op;
                        r2->consumers.push_back(op);
                        r2->params = r1->params;
                        r2->type = r1->type;
                        r2->shape = r1->shape;
                        new_input_op->outputs.push_back(r2);
                        op->inputs.push_back(r2);
                    }
                    else
                    {
                        // insert new input to loop
                        r1->consumers.push_back(src_op);
                        src_op->inputs.push_back(r1);
                        // insert new input to sub_graph
                        last_input_op_index++;
                        std::string new_input_op_name =  "pnnx_input_" + std::to_string(last_input_op_index);
                        Operator* new_input_op = sub_pnnx_graph->new_operator_after("pnnx.Input", new_input_op_name, last_input_op);
                        Operand* r2 = sub_pnnx_graph->new_operand(in->debugName());
                        r2->producer = new_input_op;
                        r2->consumers.push_back(op);
                        r2->params = r1->params;
                        r2->type = r1->type;
                        r2->shape = r1->shape;
                        new_input_op->outputs.push_back(r2);
                        op->inputs.push_back(r2);
                    }

                }
                else{
                    r->consumers.push_back(op);
                    op->inputs.push_back(r);
                }
                
               
            }

            for (int i = 0; i < (int)n->outputs().size(); i++)
            {
                const auto& on = n->output(i);
                Operand* r = sub_pnnx_graph->new_operand(on);
                r->producer = op;
                op->outputs.push_back(r);
            }

            op->params["value"] = n;

            if (op->params["value"].type == 8)
            {
                op->type = "pnnx.Attribute";

                op->params.erase("value");

                op->attrs["data"] = n->t(torch::jit::attr::value);
            }
        }
        else if (n->kind() == c10::prim::CallMethod)
        {
            auto class_type = n->input(0)->type()->cast<torch::jit::ClassType>();
            //             const std::string& name = n->s(torch::jit::attr::name);

            //             fprintf(stderr, "call %s\n", class_type->str().c_str());

            std::string name = class_type_to_names[class_type->str()];

            std::string class_type_str = torch::jit::removeTorchMangle(class_type->str());

            std::string class_type_str_no_torch_prefix = class_type_str.substr(10);

            std::string optypename = class_type_str;

            for (const auto& ow : get_global_pnnx_fuse_module_passes())
            {
                if (class_type_str != ow->match_type_str())
                    continue;

                optypename = ow->type_str();
                break;
            }

            if (optypename == class_type_str)
            {
                optypename = class_type_str_no_torch_prefix;
            }

            Operator* op = sub_pnnx_graph->new_operator(optypename, name);

            for (int i = 1; i < (int)n->inputs().size(); i++)
            {
                const auto& in = n->input(i);
                Operand* r = sub_pnnx_graph->get_operand(in->debugName());
                if (r == 0)
                {
                    Operand* r1  = 0;
                    for (const auto& graph_pair : pnnx_graph_map) {  
                        r1 = graph_pair.second->get_operand(in->debugName());
                        if(r1 != 0)
                        break;
                    }  
                
                    assert(r1 != 0 && "cur tensor name : %s not in graph\n",in->debugName().c_str()); 
                    // if(r1 == 0)
                    // {
                    //     throw std::exception("cur tensor name : %s not in graph\n",in->debugName().c_str());
                    // }
                   
                   std::string last_input_op_name =  "pnnx_input_" + std::to_string(last_input_op_index);
                    Operator* last_input_op = sub_pnnx_graph->get_operator(last_input_op_name);
                    assert(last_input_op != 0 && "failed to find last input op : %s\n",last_input_op_name.c_str());
                    if(r1->producer->type == "prim::Constant")
                    {
                        Operator* constant_op = r1->producer;
                        // insert type of prim::Constant new input to sub_graph
                        Operator* new_input_op = sub_pnnx_graph->new_operator_after(constant_op->type, constant_op->name, last_input_op);
                        new_input_op->inputnames = constant_op->inputnames;
                        new_input_op->params = constant_op->params;
                        new_input_op->attrs = constant_op->attrs;
                        Operand* r2 = sub_pnnx_graph->new_operand(in->debugName());
                        r2->producer = new_input_op;
                        r2->consumers.push_back(op);
                        r2->params = r1->params;
                        r2->type = r1->type;
                        r2->shape = r1->shape;
                        new_input_op->outputs.push_back(r2);
                        op->inputs.push_back(r2);
                    }
                    else
                    {
                        // insert new input to loop
                        r1->consumers.push_back(src_op);
                        src_op->inputs.push_back(r1);
                        // insert new input to sub_graph
                        last_input_op_index++;
                        std::string new_input_op_name =  "pnnx_input_" + std::to_string(last_input_op_index);
                        Operator* new_input_op = sub_pnnx_graph->new_operator_after("pnnx.Input", new_input_op_name, last_input_op);
                        Operand* r2 = sub_pnnx_graph->new_operand(in->debugName());
                        r2->producer = new_input_op;
                        r2->consumers.push_back(op);
                        r2->params = r1->params;
                        r2->type = r1->type;
                        r2->shape = r1->shape;
                        new_input_op->outputs.push_back(r2);
                        op->inputs.push_back(r2);
                    }

                }
                else{
                    r->consumers.push_back(op);
                    op->inputs.push_back(r);
                }
            }

            for (int i = 0; i < (int)n->outputs().size(); i++)
            {
                const auto& on = n->output(i);
                Operand* r = sub_pnnx_graph->new_operand(on);
                r->producer = op;
                op->outputs.push_back(r);
            }

            // module operator
            if (std::find(module_operators.begin(), module_operators.end(), class_type_str_no_torch_prefix) != module_operators.end())
            {
                const std::string& function_name = n->s(torch::jit::attr::name);
                torch::jit::Function& function = class_type->getMethod(function_name);
                if (function.isGraphFunction())
                {
#if TORCH_VERSION_MAJOR >= 2 || (TORCH_VERSION_MAJOR >= 1 && TORCH_VERSION_MINOR >= 11)
                    torch::jit::Block* moduleop_block = toGraphFunction(function).graph()->block();
#else
                    torch::jit::Block* moduleop_block = function.graph()->block();
#endif

                    std::map<size_t, torch::jit::Node*> constant_attr_nodes;
                    for (const auto& mn : moduleop_block->nodes())
                    {
                        if (mn->kind() == c10::prim::GetAttr)
                        {
                            std::string name = mn->s(torch::jit::attr::name);
                            //             std::string name = mn->debugName();

                            auto class_type = mn->output(0)->type()->cast<torch::jit::ClassType>();

                            if (!class_type)
                            {
                                std::deque<std::string> module_names; // = split(mn->input(0)->node()->s(torch::jit::attr::name), '.');
                                {
                                    auto np = n->input(0)->node();
                                    while (np->hasAttribute(torch::jit::attr::name))
                                    {
                                        module_names.push_front(np->s(torch::jit::attr::name));
                                        np = np->input(0)->node();
                                    }
                                }
                                std::deque<std::string> module_names2;
                                {
                                    auto np = mn->input(0)->node();
                                    while (np->hasAttribute(torch::jit::attr::name))
                                    {
                                        module_names2.push_front(np->s(torch::jit::attr::name));
                                        np = np->input(0)->node();
                                    }
                                }
                                for (auto x : module_names2)
                                {
                                    module_names.push_back(x);
                                }

                                auto sub_mod = mod;
                                for (auto module_name : module_names)
                                {
                                    sub_mod = sub_mod.attr(module_name).toModule();
                                }

                                std::string wrapped_name;
                                for (auto module_name : module_names2)
                                {
                                    if (wrapped_name.size() > 0)
                                        wrapped_name = wrapped_name + "." + module_name;
                                    else
                                        wrapped_name = module_name;
                                }

                                if (wrapped_name.empty())
                                {
                                    // top-level module
                                    wrapped_name = name;
                                }
                                else
                                {
                                    wrapped_name = wrapped_name + "." + name;
                                }

                                op->attrs[wrapped_name] = sub_mod.attr(name).toTensor();
                            }
                        }
                        else if (mn->kind() == c10::prim::Constant)
                        {
                            Parameter p(mn);

                            if (p.type == 8)
                            {
                                size_t unique_id = mn->output(0)->unique();
                                constant_attr_nodes[unique_id] = mn;
                            }
                        }
                    }

                    int pnnx_moduleop_unknown_index = 0;
                    for (auto attr : constant_attr_nodes)
                    {
                        char name[32];
                        sprintf(name, "pnnx_%02d", pnnx_moduleop_unknown_index);
                        op->attrs[name] = attr.second->t(torch::jit::attr::value);
                        pnnx_moduleop_unknown_index++;
                    }
                }
            }
            else
            {
                for (const auto& ow : get_global_pnnx_fuse_module_passes())
                {
                    if (class_type_str != ow->match_type_str())
                        continue;

                    auto class_type = n->input(0)->type()->cast<torch::jit::ClassType>();
                    torch::jit::Function& function = class_type->getMethod(n->s(torch::jit::attr::name));

                    std::deque<std::string> module_names; // = split(n->input(0)->node()->s(torch::jit::attr::name), '.');
                    {
                        auto np = n->input(0)->node();
                        while (np->hasAttribute(torch::jit::attr::name))
                        {
                            module_names.push_front(np->s(torch::jit::attr::name));
                            np = np->input(0)->node();
                        }
                    }

                    std::string wrapped_name;
                    auto sub_mod = mod;
                    for (auto module_name : module_names)
                    {
                        if (wrapped_name.size() > 0)
                            wrapped_name = wrapped_name + "." + module_name;
                        else
                            wrapped_name = module_name;
                        sub_mod = sub_mod.attr(module_name).toModule();
                    }

                    op->name = wrapped_name;

#if TORCH_VERSION_MAJOR >= 2 || (TORCH_VERSION_MAJOR >= 1 && TORCH_VERSION_MINOR >= 11)
                    ow->write(op, toGraphFunction(function).graph(), sub_mod);
#else
                    ow->write(op, function.graph(), sub_mod);
#endif

                    break;
                }
            }
        }
        // else if (n->kind() == c10::prim::CallFunction)
        // {
        //     fprintf(stderr, "function %s", n->kind().toDisplayString());
        //
        //     AT_ASSERT(cur->input(0)->node()->kind() == c10::prim::Constant);
        //     auto function_constant = cur->input(0)->node();
        //     auto fun_type = function_constant->output()->type()->expect<torch::jit::FunctionType>();
        //     if (!fun_type->function()->isGraphFunction())
        //     {
        //         continue;
        //     }
        //     cur->removeInput(0);
        //
        //     fprintf(stderr, "inline function %s\n", fun_type->function()->name().c_str());
        //
        //     GRAPH_UPDATE("Inlining function '", fun_type->function()->name(), "' to ", *cur);
        //     GRAPH_UPDATE("Function body: ", *fun_type->function()->optimized_graph());
        //     inlineCallTo(cur, fun_type->function(), false);
        //     break;
        // }
        else if(n->kind() == c10::prim::Loop)
        {
            char loop_op_name[32];
            sprintf(loop_op_name, "pnnx_loop_%d", pnnx_loop_index++);

            Operator* op = sub_pnnx_graph->new_operator(n->kind().toDisplayString(), loop_op_name);

            for (int i = 0; i < (int)n->inputs().size(); i++)
            {
                const auto& in = n->input(i);
                Operand* r =sub_pnnx_graph->get_operand(in->debugName());
                if (r == 0)
                {
                    Operand* r1  = 0;
                    for (const auto& graph_pair : pnnx_graph_map) {  
                        r1 = graph_pair.second->get_operand(in->debugName());
                        if(r1 != 0)
                        break;
                    }  
                
                    assert(r1 != 0 && "cur tensor name : %s not in graph\n",in->debugName().c_str()); 
                    // if(r1 == 0)
                    // {
                    //     throw std::exception("cur tensor name : %s not in graph\n",in->debugName().c_str());
                    // }
                   
                    std::string last_input_op_name =  "pnnx_input_" + std::to_string(last_input_op_index);
                    Operator* last_input_op = sub_pnnx_graph->get_operator(last_input_op_name);
                    assert(last_input_op != 0 && "failed to find last input op : %s\n",last_input_op_name.c_str());
                    if(r1->producer->type == "prim::Constant")
                    {
                        Operator* constant_op = r1->producer;
                        // insert type of prim::Constant new input to sub_graph
                        Operator* new_input_op = sub_pnnx_graph->new_operator_after(constant_op->type, constant_op->name, last_input_op);
                        new_input_op->inputnames = constant_op->inputnames;
                        new_input_op->params = constant_op->params;
                        new_input_op->attrs = constant_op->attrs;
                        Operand* r2 = sub_pnnx_graph->new_operand(in->debugName());
                        r2->producer = new_input_op;
                        r2->consumers.push_back(op);
                        r2->params = r1->params;
                        r2->type = r1->type;
                        r2->shape = r1->shape;
                        new_input_op->outputs.push_back(r2);
                        op->inputs.push_back(r2);
                    }
                    else
                    {
                        // insert new input to loop
                        r1->consumers.push_back(src_op);
                        src_op->inputs.push_back(r1);
                        // insert new input to sub_graph
                        last_input_op_index++;
                        std::string new_input_op_name =  "pnnx_input_" + std::to_string(last_input_op_index);
                        Operator* new_input_op = sub_pnnx_graph->new_operator_after("pnnx.Input", new_input_op_name, last_input_op);
                        Operand* r2 = sub_pnnx_graph->new_operand(in->debugName());
                        r2->producer = new_input_op;
                        r2->consumers.push_back(op);
                        r2->params = r1->params;
                        r2->type = r1->type;
                        r2->shape = r1->shape;
                        new_input_op->outputs.push_back(r2);
                        op->inputs.push_back(r2);
                    }

                }
                else{
                    r->consumers.push_back(op);
                    op->inputs.push_back(r);
                }
            }

            for (int i = 0; i < (int)n->outputs().size(); i++)
            {
                const auto& on = n->output(i);
                Operand* r = sub_pnnx_graph->new_operand(on);
                r->producer = op;
                op->outputs.push_back(r);
            }
            std::shared_ptr<Graph> sub_pnnx_graph2 = std::make_shared<Graph>();
            int block_num = 0;
            for (torch::jit::Block* subBlock2 : n->blocks())
            {
                
                assert(block_num == 0 && "block num > 1 in loop"); 
                pass_level1_block(mod, op, subBlock2, sub_pnnx_graph2, module_operators, pnnx_graph_map, pnnx_unknown_index, pnnx_loop_index);
                block_num++;
            }
            pnnx_graph_map[std::string(loop_op_name)] = sub_pnnx_graph2;
                
        }
        else
        {
            char name[32];
            sprintf(name, "pnnx_%d", pnnx_unknown_index++);

            Operator* op = sub_pnnx_graph->new_operator(n->kind().toDisplayString(), name);

            for (int i = 0; i < (int)n->inputs().size(); i++)
            {
                const auto& in = n->input(i);
                Operand* r = sub_pnnx_graph->get_operand(in->debugName());
                if (r == 0)
                {
                    Operand* r1  = 0;
                    for (const auto& graph_pair : pnnx_graph_map) {  
                        r1 = graph_pair.second->get_operand(in->debugName());
                        if(r1 != 0)
                        break;
                    }  
                
                    assert(r1 != 0 && "cur tensor name : %s not in graph\n",in->debugName().c_str());  
                    // if(r1 == 0)
                    // {
                    //     throw std::exception("cur tensor name : %s not in graph\n",in->debugName().c_str());
                    // }
                   
                    std::string last_input_op_name =  "pnnx_input_" + std::to_string(last_input_op_index);
                    Operator* last_input_op = sub_pnnx_graph->get_operator(last_input_op_name);
                    assert(last_input_op != 0 && "failed to find last input op : %s\n",last_input_op_name.c_str());
                    if(r1->producer->type == "prim::Constant")
                    {
                        Operator* constant_op = r1->producer;
                        // insert type of prim::Constant new input to sub_graph
                        Operator* new_input_op = sub_pnnx_graph->new_operator_after(constant_op->type, constant_op->name, last_input_op);
                        new_input_op->inputnames = constant_op->inputnames;
                        new_input_op->params = constant_op->params;
                        new_input_op->attrs = constant_op->attrs;
                        Operand* r2 = sub_pnnx_graph->new_operand(in->debugName());
                        r2->producer = new_input_op;
                        r2->consumers.push_back(op);
                        r2->params = r1->params;
                        r2->type = r1->type;
                        r2->shape = r1->shape;
                        new_input_op->outputs.push_back(r2);
                        op->inputs.push_back(r2);
                    }
                    else
                    {
                        // insert new input to loop
                        r1->consumers.push_back(src_op);
                        src_op->inputs.push_back(r1);
                        // insert new input to sub_graph
                        last_input_op_index++;
                        std::string new_input_op_name =  "pnnx_input_" + std::to_string(last_input_op_index);
                        Operator* new_input_op = sub_pnnx_graph->new_operator_after("pnnx.Input", new_input_op_name, last_input_op);
                        Operand* r2 = sub_pnnx_graph->new_operand(in->debugName());
                        r2->producer = new_input_op;
                        r2->consumers.push_back(op);
                        r2->params = r1->params;
                        r2->type = r1->type;
                        r2->shape = r1->shape;
                        new_input_op->outputs.push_back(r2);
                        op->inputs.push_back(r2);
                    }

                }
                else{
                    r->consumers.push_back(op);
                    op->inputs.push_back(r);
                }
            }

            for (int i = 0; i < (int)n->outputs().size(); i++)
            {
                const auto& on = n->output(i);
                Operand* r = sub_pnnx_graph->new_operand(on);
                r->producer = op;
                op->outputs.push_back(r);
            }

        }
    }

    for (int i = 1; i < (int)sub_block->outputs().size(); i++)
    {
        const auto& out = sub_block->outputs()[i];
        char output_name[32];
        sprintf(output_name, "pnnx_output_%d", i - 1);
        // out->debugName()
        Operator* op = sub_pnnx_graph->new_operator("pnnx.Output", output_name);
        Operand* r = sub_pnnx_graph->get_operand(out->debugName());
        Operand* src_r =  src_op->outputs.at(i - 1);
        r->params = src_r->params;
        r->type = src_r->type;
        r->shape = src_r->shape;
        r->consumers.push_back(op);
        op->inputs.push_back(r);
    }
        // post process
    fuse_moduleop_unpack(sub_pnnx_graph, module_operators);


}

void pass_level1(const torch::jit::Module& mod,\
                const std::shared_ptr<torch::jit::Graph>& g, \
                const std::vector<std::string>& module_operators,\
                std::unordered_map<std::string, std::shared_ptr<pnnx::Graph>>& pnnx_graph_map)
{
    std::shared_ptr<Graph> pg = std::make_shared<Graph>();
    pnnx_graph_map["src"] = pg;
    for (int i = 1; i < (int)g->inputs().size(); i++)
    {
        const auto& in = g->inputs()[i];

        char name[32];
        sprintf(name, "pnnx_input_%d", i - 1);

        Operator* op = pg->new_operator("pnnx.Input", name);
        Operand* r = pg->new_operand(in);
        r->producer = op;
        op->outputs.push_back(r);
    }

    std::map<std::string, std::string> class_type_to_names;
    int pnnx_unknown_index = 0;
    int pnnx_loop_index = 0;
    for (const auto& n : g->block()->nodes())
    {
        if (n->kind() == c10::prim::GetAttr)
        {
            // pass
            std::string name = n->s(torch::jit::attr::name);
            //             std::string name = n->debugName();

            auto class_type = n->output(0)->type()->cast<torch::jit::ClassType>();

            if (class_type)
            {
                std::string class_type_str = class_type->str();
                class_type_to_names[class_type_str] = name;
                //             class_type_to_names[class_type_str] = class_type_str + "." + name;
            }
            else
            {
                // Tensor from some class
                //                 Operator* op = pg->new_operator(n->kind().toDisplayString(), name);
                Operator* op = pg->new_operator("pnnx.Attribute", name);

                for (int i = 0; i < (int)n->outputs().size(); i++)
                {
                    const auto& on = n->output(i);
                    Operand* r = pg->new_operand(on);
                    r->producer = op;
                    op->outputs.push_back(r);
                }

                std::deque<std::string> module_names; // = split(n->input(0)->node()->s(torch::jit::attr::name), '.');
                {
                    auto np = n->input(0)->node();
                    while (np->hasAttribute(torch::jit::attr::name))
                    {
                        module_names.push_front(np->s(torch::jit::attr::name));
                        np = np->input(0)->node();
                    }
                }

                std::string wrapped_name;
                auto sub_mod = mod;
                for (auto module_name : module_names)
                {
                    if (wrapped_name.size() > 0)
                        wrapped_name = wrapped_name + "." + module_name;
                    else
                        wrapped_name = module_name;
                    sub_mod = sub_mod.attr(module_name).toModule();
                }

                if (wrapped_name.empty())
                {
                    // top-level module
                    wrapped_name = name;
                }

                op->name = wrapped_name;

                // op->params["this"] = n->input(i)

                // sub_mod.dump(true, true, true);

                op->attrs["data"] = sub_mod.attr(name).toTensor();
                op->outputs[0]->type = op->attrs["data"].type;
                op->outputs[0]->shape = op->attrs["data"].shape;
            }
        }
        else if (n->kind() == c10::prim::Constant) // || n->kind() == c10::prim::ListConstruct)
        {
            char name[32];
            sprintf(name, "pnnx_%d", pnnx_unknown_index++);

            Operator* op = pg->new_operator(n->kind().toDisplayString(), name);

            for (int i = 0; i < (int)n->inputs().size(); i++)
            {
                const auto& in = n->input(i);
                Operand* r = pg->get_operand(in->debugName());
                r->consumers.push_back(op);
                op->inputs.push_back(r);
            }

            for (int i = 0; i < (int)n->outputs().size(); i++)
            {
                const auto& on = n->output(i);
                Operand* r = pg->new_operand(on);
                r->producer = op;
                op->outputs.push_back(r);
            }

            op->params["value"] = n;

            if (op->params["value"].type == 8)
            {
                op->type = "pnnx.Attribute";

                op->params.erase("value");

                op->attrs["data"] = n->t(torch::jit::attr::value);
            }
        }
        else if (n->kind() == c10::prim::CallMethod)
        {
            auto class_type = n->input(0)->type()->cast<torch::jit::ClassType>();
            //             const std::string& name = n->s(torch::jit::attr::name);

            //             fprintf(stderr, "call %s\n", class_type->str().c_str());

            std::string name = class_type_to_names[class_type->str()];

            std::string class_type_str = torch::jit::removeTorchMangle(class_type->str());

            std::string class_type_str_no_torch_prefix = class_type_str.substr(10);

            std::string optypename = class_type_str;

            for (const auto& ow : get_global_pnnx_fuse_module_passes())
            {
                if (class_type_str != ow->match_type_str())
                    continue;

                optypename = ow->type_str();
                break;
            }

            if (optypename == class_type_str)
            {
                optypename = class_type_str_no_torch_prefix;
            }

            Operator* op = pg->new_operator(optypename, name);

            for (int i = 1; i < (int)n->inputs().size(); i++)
            {
                const auto& in = n->input(i);
                Operand* r = pg->get_operand(in->debugName());
                r->consumers.push_back(op);
                op->inputs.push_back(r);
            }

            for (int i = 0; i < (int)n->outputs().size(); i++)
            {
                const auto& on = n->output(i);
                Operand* r = pg->new_operand(on);
                r->producer = op;
                op->outputs.push_back(r);
            }

            // module operator
            if (std::find(module_operators.begin(), module_operators.end(), class_type_str_no_torch_prefix) != module_operators.end())
            {
                const std::string& function_name = n->s(torch::jit::attr::name);
                torch::jit::Function& function = class_type->getMethod(function_name);
                if (function.isGraphFunction())
                {
#if TORCH_VERSION_MAJOR >= 2 || (TORCH_VERSION_MAJOR >= 1 && TORCH_VERSION_MINOR >= 11)
                    torch::jit::Block* moduleop_block = toGraphFunction(function).graph()->block();
#else
                    torch::jit::Block* moduleop_block = function.graph()->block();
#endif

                    std::map<size_t, torch::jit::Node*> constant_attr_nodes;
                    for (const auto& mn : moduleop_block->nodes())
                    {
                        if (mn->kind() == c10::prim::GetAttr)
                        {
                            std::string name = mn->s(torch::jit::attr::name);
                            //             std::string name = mn->debugName();

                            auto class_type = mn->output(0)->type()->cast<torch::jit::ClassType>();

                            if (!class_type)
                            {
                                std::deque<std::string> module_names; // = split(mn->input(0)->node()->s(torch::jit::attr::name), '.');
                                {
                                    auto np = n->input(0)->node();
                                    while (np->hasAttribute(torch::jit::attr::name))
                                    {
                                        module_names.push_front(np->s(torch::jit::attr::name));
                                        np = np->input(0)->node();
                                    }
                                }
                                std::deque<std::string> module_names2;
                                {
                                    auto np = mn->input(0)->node();
                                    while (np->hasAttribute(torch::jit::attr::name))
                                    {
                                        module_names2.push_front(np->s(torch::jit::attr::name));
                                        np = np->input(0)->node();
                                    }
                                }
                                for (auto x : module_names2)
                                {
                                    module_names.push_back(x);
                                }

                                auto sub_mod = mod;
                                for (auto module_name : module_names)
                                {
                                    sub_mod = sub_mod.attr(module_name).toModule();
                                }

                                std::string wrapped_name;
                                for (auto module_name : module_names2)
                                {
                                    if (wrapped_name.size() > 0)
                                        wrapped_name = wrapped_name + "." + module_name;
                                    else
                                        wrapped_name = module_name;
                                }

                                if (wrapped_name.empty())
                                {
                                    // top-level module
                                    wrapped_name = name;
                                }
                                else
                                {
                                    wrapped_name = wrapped_name + "." + name;
                                }

                                op->attrs[wrapped_name] = sub_mod.attr(name).toTensor();
                            }
                        }
                        else if (mn->kind() == c10::prim::Constant)
                        {
                            Parameter p(mn);

                            if (p.type == 8)
                            {
                                size_t unique_id = mn->output(0)->unique();
                                constant_attr_nodes[unique_id] = mn;
                            }
                        }
                    }

                    int pnnx_moduleop_unknown_index = 0;
                    for (auto attr : constant_attr_nodes)
                    {
                        char name[32];
                        sprintf(name, "pnnx_%02d", pnnx_moduleop_unknown_index);
                        op->attrs[name] = attr.second->t(torch::jit::attr::value);
                        pnnx_moduleop_unknown_index++;
                    }
                }
            }
            else
            {
                for (const auto& ow : get_global_pnnx_fuse_module_passes())
                {
                    if (class_type_str != ow->match_type_str())
                        continue;

                    auto class_type = n->input(0)->type()->cast<torch::jit::ClassType>();
                    torch::jit::Function& function = class_type->getMethod(n->s(torch::jit::attr::name));

                    std::deque<std::string> module_names; // = split(n->input(0)->node()->s(torch::jit::attr::name), '.');
                    {
                        auto np = n->input(0)->node();
                        while (np->hasAttribute(torch::jit::attr::name))
                        {
                            module_names.push_front(np->s(torch::jit::attr::name));
                            np = np->input(0)->node();
                        }
                    }

                    std::string wrapped_name;
                    auto sub_mod = mod;
                    for (auto module_name : module_names)
                    {
                        if (wrapped_name.size() > 0)
                            wrapped_name = wrapped_name + "." + module_name;
                        else
                            wrapped_name = module_name;
                        sub_mod = sub_mod.attr(module_name).toModule();
                    }

                    op->name = wrapped_name;

#if TORCH_VERSION_MAJOR >= 2 || (TORCH_VERSION_MAJOR >= 1 && TORCH_VERSION_MINOR >= 11)
                    ow->write(op, toGraphFunction(function).graph(), sub_mod);
#else
                    ow->write(op, function.graph(), sub_mod);
#endif

                    break;
                }
            }
        }
        // else if (n->kind() == c10::prim::CallFunction)
        // {
        //     fprintf(stderr, "function %s", n->kind().toDisplayString());
        //
        //     AT_ASSERT(cur->input(0)->node()->kind() == c10::prim::Constant);
        //     auto function_constant = cur->input(0)->node();
        //     auto fun_type = function_constant->output()->type()->expect<torch::jit::FunctionType>();
        //     if (!fun_type->function()->isGraphFunction())
        //     {
        //         continue;
        //     }
        //     cur->removeInput(0);
        //
        //     fprintf(stderr, "inline function %s\n", fun_type->function()->name().c_str());
        //
        //     GRAPH_UPDATE("Inlining function '", fun_type->function()->name(), "' to ", *cur);
        //     GRAPH_UPDATE("Function body: ", *fun_type->function()->optimized_graph());
        //     inlineCallTo(cur, fun_type->function(), false);
        //     break;
        // }
        else if(n->kind() == c10::prim::Loop)
        {
            char loop_op_name[32];
            sprintf(loop_op_name, "pnnx_loop_%d", pnnx_loop_index++);

            Operator* op = pg->new_operator(n->kind().toDisplayString(), loop_op_name);

            for (int i = 0; i < (int)n->inputs().size(); i++)
            {
                const auto& in = n->input(i);
                Operand* r = pg->get_operand(in->debugName());
                r->consumers.push_back(op);
                op->inputs.push_back(r);
            }

            for (int i = 0; i < (int)n->outputs().size(); i++)
            {
                const auto& on = n->output(i);
                Operand* r = pg->new_operand(on);
                r->producer = op;
                op->outputs.push_back(r);
            }

            std::shared_ptr<Graph> sub_pnnx_graph = std::make_shared<Graph>();
            int block_num = 0;
            for (torch::jit::Block* subBlock : n->blocks())
            {
                
                assert(block_num == 0 && "block num > 1 in loop"); 
                pass_level1_block(mod, op, subBlock, sub_pnnx_graph, module_operators, pnnx_graph_map, pnnx_unknown_index, pnnx_loop_index);
                block_num++;
            }
            pnnx_graph_map[std::string(loop_op_name)] = sub_pnnx_graph;
                
        }
        else
        {
            char name[32];
            sprintf(name, "pnnx_%d", pnnx_unknown_index++);

            Operator* op = pg->new_operator(n->kind().toDisplayString(), name);

            for (int i = 0; i < (int)n->inputs().size(); i++)
            {
                const auto& in = n->input(i);
                Operand* r = pg->get_operand(in->debugName());
                r->consumers.push_back(op);
                op->inputs.push_back(r);
            }

            for (int i = 0; i < (int)n->outputs().size(); i++)
            {
                const auto& on = n->output(i);
                Operand* r = pg->new_operand(on);
                r->producer = op;
                op->outputs.push_back(r);
            }

        }
    }

    for (int i = 0; i < (int)g->outputs().size(); i++)
    {
        const auto& in = g->outputs()[i];

        char name[32];
        sprintf(name, "pnnx_output_%d", i);
        Operator* op = pg->new_operator("pnnx.Output", name);
        Operand* r = pg->get_operand(in->debugName());
        r->consumers.push_back(op);
        op->inputs.push_back(r);
    }

    // post process
    fuse_moduleop_unpack(pg, module_operators);
}

} // namespace pnnx
