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

#define ASSERT_INFO(expr, info) \
do { \
    if (!(expr)) { \
        std::cerr << "Assertion failed: " << #expr << "\n" \
                  << "Info: " << info << "\n"; \
        assert(0); \
    } \
} while (0)

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


void PassLevel1::Process(const torch::jit::Module& mod,
                const std::shared_ptr<torch::jit::Graph>& g, 
                const std::vector<std::string>& module_operators,
                std::shared_ptr<pnnx::MainGraph>& pnnx_graph)
{
    // create main graph
    std::string main_graph_name = "src";
    pnnx_graph->create_main_graph(main_graph_name);
    std::shared_ptr<pnnx::Graph> pg = pnnx_graph->get_main_graph();
    this->_module_operators = module_operators;

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

    for (const auto& n : g->block()->nodes())
    {
        if (n->kind() == c10::prim::GetAttr)
        {
            
           this->Process_GetAttr(mod, n, pg);
        }
        else if (n->kind() == c10::prim::Constant) // || n->kind() == c10::prim::ListConstruct)
        {
           this->Process_Constant(n, pnnx_graph);
        }
        else if (n->kind() == c10::prim::CallMethod)
        {
           this->Process_CallMethod(mod, n, pnnx_graph);
        }
    
        else if(n->kind() == c10::prim::Loop)
        {
            this->Process_Loop(mod, n, pnnx_graph);    
        }
        else if(n->kind() == c10::prim::If)
        {
           this->Process_If(mod, n, pnnx_graph);      
        }
        else
        {
           this->Process_Other(n, pnnx_graph);
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
    fuse_moduleop_unpack(pg, _module_operators);
}

void PassLevel1::Process_GetAttr(const torch::jit::Module& mod, const torch::jit::Node* n, std::shared_ptr<pnnx::Graph>& pg)
{
     // pass
    std::string name = n->s(torch::jit::attr::name);
    //             std::string name = n->debugName();

    auto class_type = n->output(0)->type()->cast<torch::jit::ClassType>();

    if (class_type)
    {
        std::string class_type_str = class_type->str();
        this->class_type_to_names[class_type_str] = name;
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

void PassLevel1::Process_Constant(
            const torch::jit::Node* n, 
            std::shared_ptr<pnnx::MainGraph>& pnnx_graph)
{
    char name[32];
    sprintf(name, "pnnx_%d", this->pnnx_unknown_index++);

    // create op
    std::shared_ptr<pnnx::Graph> pg = pnnx_graph->get_main_graph();

    Operator* op = pg->new_operator(n->kind().toDisplayString(), name);

    this->Process_Input(op, pnnx_graph, n);
    this->Process_Output(op, n, pg);

    op->params["value"] = n;

    if (op->params["value"].type == 8)
    {
        op->type = "pnnx.Attribute";

        op->params.erase("value");

        op->attrs["data"] = n->t(torch::jit::attr::value);
    }
}

void PassLevel1::Process_CallMethod(const torch::jit::Module& mod, 
            const torch::jit::Node* n, 
            std::shared_ptr<pnnx::MainGraph>& pnnx_graph)
{
     auto class_type = n->input(0)->type()->cast<torch::jit::ClassType>();
    //             const std::string& name = n->s(torch::jit::attr::name);

    //             fprintf(stderr, "call %s\n", class_type->str().c_str());

    std::string name = this->class_type_to_names[class_type->str()];

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

     // create op
    std::shared_ptr<pnnx::Graph> pg = pnnx_graph->get_main_graph();
    Operator* op = pg->new_operator(optypename, name);
    
    this->Process_Input(op, pnnx_graph, n, 1);
    this->Process_Output(op, n, pg);

    // module operator
    if (std::find(_module_operators.begin(), _module_operators.end(), class_type_str_no_torch_prefix) != _module_operators.end())
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


void PassLevel1::Process_Loop(const torch::jit::Module& mod, 
            const torch::jit::Node* n, 
            std::shared_ptr<pnnx::MainGraph>& pnnx_graph)
{
    char loop_op_name[32];
    sprintf(loop_op_name, "pnnx_loop_%d", pnnx_loop_index++);
    // create op
    std::shared_ptr<pnnx::Graph> pg = pnnx_graph->get_main_graph();
    Operator* op = pg->new_operator(n->kind().toDisplayString(), loop_op_name);
    
    this->Process_Input(op, pnnx_graph, n);
    this->Process_Output(op, n, pg);
    
    std::vector<std::string> block_names;     
    int block_num = 0;
    int init_input_num = op->inputs.size();
    for (const torch::jit::Block* subBlock : n->blocks())
    {
        std::string loop_block_name = std::string(loop_op_name);
        std::shared_ptr<pnnx::MainGraph> sub_pnnx_graph = std::make_shared<pnnx::MainGraph>();
        sub_pnnx_graph->create_main_graph(loop_block_name);
        pnnx_graph->insert_sub_graph(loop_block_name, sub_pnnx_graph, op, init_input_num);
        sub_pnnx_graph->set_base_graph(pnnx_graph);
        // ASSERT_INFO(block_num == 0 , "block num > 1 in loop"); 
        
        Process_Loop_block(mod, subBlock, sub_pnnx_graph, op, loop_block_name);
        pnnx_graph->effective_sub_model_name.push_back(loop_block_name);
        block_num++;
        block_names.push_back(loop_block_name);

    }
    
   
}


void PassLevel1::Process_If(const torch::jit::Module& mod, 
            const torch::jit::Node* n, 
            std::shared_ptr<pnnx::MainGraph>& pnnx_graph)
{

    char if_op_name[32];
    sprintf(if_op_name, "pnnx_if_%d", pnnx_if_index++);

    // create op
    std::shared_ptr<pnnx::Graph> pg = pnnx_graph->get_main_graph();
    Operator* op = pg->new_operator(n->kind().toDisplayString(), if_op_name);

    this->Process_Input(op, pnnx_graph, n);
    this->Process_Output(op, n, pg);

    std::vector<std::string> block_names;
    int block_index = 0;
    // ASSERT_INFO(subBlock_num != 2 && "block num eq 2 in if"); 
    for (const torch::jit::Block* subBlock : n->blocks())
    {
        if(subBlock->nodes().front() == subBlock->nodes().back())
        {
            continue;
        }
        std::string sub_if_op_block_name = std::string(if_op_name) + "_block" + std::to_string(block_index);
        
        std::shared_ptr<pnnx::MainGraph> sub_pnnx_graph = std::make_shared<pnnx::MainGraph>();
        sub_pnnx_graph->create_main_graph(sub_if_op_block_name);
        pnnx_graph->insert_sub_graph(sub_if_op_block_name, sub_pnnx_graph, op);
        sub_pnnx_graph->set_base_graph(pnnx_graph);

        Process_If_block(mod, subBlock, sub_pnnx_graph, sub_if_op_block_name);
        pnnx_graph->effective_sub_model_name.push_back(sub_if_op_block_name);
        block_names.push_back(sub_if_op_block_name);
        block_index++;
    }
}

void PassLevel1::Process_If_block(const torch::jit::Module& mod, 
    const torch::jit::Block* sub_block, 
    std::shared_ptr<pnnx::MainGraph>& pnnx_graph,
    std::string& sub_op_block_name)
{
    //
   std::shared_ptr<pnnx::Graph> pg = pnnx_graph->get_main_graph();
    // loop nodes in cur block
    for (const auto& n : sub_block->nodes())
    {
        if (n->kind() == c10::prim::GetAttr)
        {
            this->Process_GetAttr(mod, n, pg);
        }
        else if (n->kind() == c10::prim::Constant) // || n->kind() == c10::prim::ListConstruct)
        {
             this->Process_Constant(n, pnnx_graph);
        }
        else if (n->kind() == c10::prim::CallMethod)
        {
           this->Process_CallMethod(mod, n, pnnx_graph);
        }
        // else if (n->kind() == c10::prim::CallFunction)
        // {
        //     fprintf(stderr, "function %s", n->kind().toDisplayString());

        //     // AT_ASSERT(n->input(0)->node()->kind() == c10::prim::Constant);
        //     auto function_constant = n->input(0)->node();
        //     auto fun_type = function_constant->output()->type()->expect<torch::jit::FunctionType>();
        //     if (!fun_type->function()->isGraphFunction())
        //     {
        //         continue;
        //     }
        //     // n->removeInput(0);
        
        //     fprintf(stderr, "inline function %s\n", fun_type->function()->name().c_str());
        
        //     // GRAPH_UPDATE("Inlining function '", fun_type->function()->name(), "' to ", *n);
        //     // GRAPH_UPDATE("Function body: ", *fun_type->function()->optimized_graph());
        //     // inlineCallTo(n, fun_type->function(), false);
        //     // break;
        // }
        else if(n->kind() == c10::prim::Loop)
        {
            this->Process_Loop(mod, n, pnnx_graph);  
                
        }
        else if(n->kind() == c10::prim::If)
        {
             this->Process_If(mod, n, pnnx_graph);
        }
        else
        {
             this->Process_Other(n, pnnx_graph);
        }
    }

    // get_base_op
    // Operator* base_op = pnnx_graph->get_base_op(sub_op_block_name);
    for (int i = 0; i < (int)sub_block->outputs().size(); i++)
    {
        const auto& out = sub_block->outputs()[i];
        char output_name[32];
        sprintf(output_name, "pnnx_output_%d", i);
        // out->debugName()
        Operator* op = pg->new_operator("pnnx.Output", output_name);
        Operand* r = pg->get_operand(out->debugName());
        // get_base_op 
        // Operand* src_r =  base_op->outputs.at(i);
        // r->params = src_r->params;
        // r->type = src_r->type;
        // r->shape = src_r->shape;
        r->consumers.push_back(op);
        op->inputs.push_back(r);
    }
        // post process
    fuse_moduleop_unpack(pg,  _module_operators);
}



void PassLevel1::Process_Other(const torch::jit::Node* n, 
     std::shared_ptr<pnnx::MainGraph>& pnnx_graph)
{
    char name[32];
    sprintf(name, "pnnx_%d", pnnx_unknown_index++);
    // create op
    std::shared_ptr<pnnx::Graph> pg = pnnx_graph->get_main_graph();
    Operator* op = pg->new_operator(n->kind().toDisplayString(), name);
    this->Process_Input(op, pnnx_graph, n);
    this->Process_Output(op, n, pg);
}

void PassLevel1::Process_Loop_block(const torch::jit::Module& mod, 
    const torch::jit::Block* sub_block, 
    std::shared_ptr<pnnx::MainGraph>& pnnx_graph,
    Operator* loop_op,
    std::string& sub_op_block_name)
{
    std::shared_ptr<pnnx::Graph> pg = pnnx_graph->get_main_graph();
    // create_input
    for (int i = 1; i < (int)sub_block->inputs().size(); i++)
    {
        char input_name[32];
        sprintf(input_name, "pnnx_input_%d", i - 1);
        const auto& block_input = sub_block->inputs()[i];
        // block_input->debugName()
        Operator* op = pg->new_operator("pnnx.Input", std::string(input_name));
        Operand* r = pg->new_operand(block_input);
        Operand* src_r =  loop_op->inputs.at(i + 1);
        r->params = src_r->params;
        r->type = src_r->type;
        r->shape = src_r->shape;
        r->producer = op;
        op->outputs.push_back(r);
    }
   
    for (const auto& n : sub_block->nodes())
    {
        if (n->kind() == c10::prim::GetAttr)
        {
            this->Process_GetAttr(mod, n, pg);
           
        }
        else if (n->kind() == c10::prim::Constant) // || n->kind() == c10::prim::ListConstruct)
        {
             this->Process_Constant(n, pnnx_graph);
           
        }
        else if (n->kind() == c10::prim::CallMethod)
        {
           this->Process_CallMethod(mod, n, pnnx_graph);
        }
        else if(n->kind() == c10::prim::Loop)
        {
           this->Process_Loop(mod, n, pnnx_graph);   
        }
        else if(n->kind() == c10::prim::If)
        {
            this->Process_If(mod, n, pnnx_graph);
        }
        else
        {
            this->Process_Other(n, pnnx_graph);
        }
    }

    for (int i = 1; i < (int)sub_block->outputs().size(); i++)
    {
        const auto& out = sub_block->outputs()[i];
        char output_name[32];
        sprintf(output_name, "pnnx_output_%d", i - 1);
        // out->debugName()
        Operator* op = pg->new_operator("pnnx.Output", output_name);
        Operand* r = pg->get_operand(out->debugName());
        Operand* src_r =  loop_op->outputs.at(i - 1);
        r->params = src_r->params;
        r->type = src_r->type;
        r->shape = src_r->shape;
        r->consumers.push_back(op);
        op->inputs.push_back(r);
    }
        // post process
    fuse_moduleop_unpack(pg, _module_operators); 
    // src_op->params[sub_op_block_name + "_inputs_index"] = sub_op_input_index;
}




void PassLevel1::Process_Input(Operator* op, 
    std::shared_ptr<pnnx::MainGraph>& pnnx_graph,
    const torch::jit::Node* n,
    int start_index)
{
    // get cur pg
    std::shared_ptr<pnnx::Graph> cur_pg = pnnx_graph->get_main_graph();
    // 
    for (int i = start_index; i < (int)n->inputs().size(); i++)
    {
        const auto& in = n->input(i);
        
        Operand* r = cur_pg->get_operand(in->debugName());
        if (r == 0)
        {
            Operand* r1  = 0;
            //search in->debugName() from base graph
            std::vector<std::string> base_graph_names; 
            std::shared_ptr<pnnx::MainGraph> tmp_base_graph = pnnx_graph;
            base_graph_names.push_back(tmp_base_graph->get_pnnx_graph_name());
            while(tmp_base_graph->get_base_graph() != nullptr)
            {
                auto base_graph = tmp_base_graph->get_base_graph();
                auto base_main_graph = base_graph->get_main_graph();
                r1 = base_main_graph->get_operand(in->debugName());
                if(r1 != 0)
                break;
                tmp_base_graph = base_graph;
                base_graph_names.push_back(tmp_base_graph->get_pnnx_graph_name());
            }
           
            if(r1->producer->type == "prim::Constant")
            {
                
                Operator* constant_op = r1->producer;
                Operator* new_input_op;
                new_input_op = cur_pg->new_constant_operator(constant_op->type, constant_op->name);
                // int cur_op_last_input_op_index = op->inputs.size() -1;
                // if(cur_op_last_input_op_index == -1)
                // {
                //     new_input_op = cur_pg->new_operator(constant_op->type, constant_op->name);
                // }
                // else
                // {
                //     std::string last_input_op_name =  "pnnx_input_" + std::to_string(cur_op_last_input_op_index);
                //     Operator* last_input_op = cur_pg->get_operator(last_input_op_name);
                //     new_input_op = cur_pg->new_operator_after(constant_op->type, constant_op->name, last_input_op);
                // }
                // insert type of prim::Constant new input to sub_graph
                new_input_op->inputnames = constant_op->inputnames;
                new_input_op->params = constant_op->params;
                new_input_op->attrs = constant_op->attrs;
                Operand* r2 = cur_pg->new_operand(in->debugName());
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
                // set this op with new input
                auto base_graph = tmp_base_graph->get_base_graph();
                std::string sub_graph_name = base_graph_names.back();
                for (auto it = base_graph->op_2_graph.begin(); it != base_graph->op_2_graph.end(); ++it) 
                {  
                    auto& op_2_graph_input_list = it->second;
                    for (auto it2 = op_2_graph_input_list.begin(); it2 != op_2_graph_input_list.end(); ++it2) 
                    {
                        if(it2->first == sub_graph_name)
                        {
                            // get op 
                            auto base_main_graph = base_graph->get_main_graph();
                            Operator* src_op = base_main_graph->get_operator(it->first);
                            auto input_it = std::find(src_op->inputs.begin(), src_op->inputs.end(), r1);
                            if(input_it == src_op->inputs.end())
                            {
                                r1->consumers.push_back(src_op);
                                src_op->inputs.push_back(r1);
                                int last_src_input_index = src_op->inputs.size() -1; 
                                it2->second.push_back(last_src_input_index);
                            }
                        }
                    }  
                }

                Operator* new_input_op;
                while( base_graph_names.size()>0)
                {
                    sub_graph_name = base_graph_names.back();
                    base_graph_names.pop_back();
                    // set sub_graph new input
                    new_input_op = base_graph->set_sub_graph_new_input(sub_graph_name,in->debugName(),r1);
                    if(base_graph_names.size() > 0)
                    {
                        std::string sub_graph_name2 = base_graph_names.back();
                        // 
                        base_graph = base_graph->get_sub_graph(sub_graph_name);
                        base_graph->set_op_new_input(sub_graph_name2, new_input_op);
                    }
                   
                }
                base_graph = base_graph->get_sub_graph(sub_graph_name);
                auto base_main_graph = base_graph->get_main_graph();
                Operand* r2 = base_main_graph->get_operand(in->debugName());
                // Operand* r2 = base_main_graph->new_operand(in->debugName());
                // r2->producer = new_input_op;
                r2->consumers.push_back(op);
                // r2->params = r1->params;
                // r2->type = r1->type;
                // r2->shape = r1->shape;
                // new_input_op->outputs.push_back(r2);
                op->inputs.push_back(r2);
                
            }

        }
        else
        {
            r->consumers.push_back(op);
            op->inputs.push_back(r);
        }
    }
}

void PassLevel1::Process_Output(Operator* op, const torch::jit::Node* n, std::shared_ptr<pnnx::Graph>& pg)
{
    for (int i = 0; i < (int)n->outputs().size(); i++)
    {
        const auto& on = n->output(i);
        Operand* r = pg->new_operand(on);
        r->producer = op;
        op->outputs.push_back(r);
    }
}
} // namespace pnnx
