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

#ifndef PNNX_PASS_LEVEL1_H
#define PNNX_PASS_LEVEL1_H
#include <cassert>  
#include <unordered_map>
#include <torch/script.h>
#include <torch/csrc/jit/api/module.h>
#include "ir.h"

namespace pnnx {

class FuseModulePass
{
public:
    virtual ~FuseModulePass();

    virtual const char* match_type_str() const = 0;

    virtual const char* type_str() const = 0;

    virtual void write(Operator* op, const std::shared_ptr<torch::jit::Graph>& graph) const;

    virtual void write(Operator* op, const std::shared_ptr<torch::jit::Graph>& graph, const torch::jit::Module& mod) const;
};

class FuseModulePassRegister
{
public:
    FuseModulePassRegister(const FuseModulePass* pass);
    ~FuseModulePassRegister();
    const FuseModulePass* pass;
};

const std::vector<const FuseModulePass*>& get_global_pnnx_fuse_module_passes();

#define REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(CLASS) \
    static FuseModulePassRegister g_global_pnnx_fusemodulepass_##CLASS##_register(new CLASS);

class PassLevel1
{
public:
    void Process(const torch::jit::Module& mod,
                const std::shared_ptr<torch::jit::Graph>& g, 
                const std::vector<std::string>& module_operators,
                std::shared_ptr<pnnx::MainGraph>& pnnx_graph);
private:

   
    void Process_GetAttr(const torch::jit::Module& mod, const torch::jit::Node* n, std::shared_ptr<pnnx::Graph>& pg);
   
    void Process_Constant(
            const torch::jit::Node* n, 
            std::shared_ptr<pnnx::MainGraph>& pnnx_graph);

    void Process_CallMethod(const torch::jit::Module& mod, 
            const torch::jit::Node* n, 
            std::shared_ptr<pnnx::MainGraph>& pnnx_graph);

    void Process_Loop(const torch::jit::Module& mod, 
            const torch::jit::Node* n, 
            std::shared_ptr<pnnx::MainGraph>& pnnx_graph);

    void Process_If(const torch::jit::Module& mod, 
            const torch::jit::Node* n, 
            std::shared_ptr<pnnx::MainGraph>& pnnx_graph);

    void Process_Other(const torch::jit::Node* n, 
     std::shared_ptr<pnnx::MainGraph>& pnnx_graph);
    
    void Process_Loop_block(const torch::jit::Module& mod, 
    const torch::jit::Block* sub_block, 
    std::shared_ptr<pnnx::MainGraph>& pnnx_graph,
    Operator* loop_op,
    std::string& sub_op_block_name);

    void Process_If_block(const torch::jit::Module& mod, 
    const torch::jit::Block* sub_block, 
    std::shared_ptr<pnnx::MainGraph>& pnnx_graph,
    std::string& sub_op_block_name);

    void Process_Input(Operator* op, 
    std::shared_ptr<pnnx::MainGraph>& pnnx_graph, 
    const torch::jit::Node* n,
    int start_index=0);

    void Process_Output(Operator* op, const torch::jit::Node* n, std::shared_ptr<pnnx::Graph>& pg);

private:
    std::map<std::string, std::string> class_type_to_names;
    int pnnx_unknown_index = 0;
    int pnnx_loop_index = 0;
    int pnnx_if_index = 0;
    std::vector<std::string> _module_operators;

};
} // namespace pnnx

#endif // PNNX_PASS_LEVEL1_H
