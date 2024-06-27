// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
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

#ifndef PNNX_LOAD_TORCHSCRIPT_H
#define PNNX_LOAD_TORCHSCRIPT_H
#include <unordered_map> 
#include <memory>
#include "ir.h"

namespace pnnx {

int load_torchscript(const std::string& ptpath, 
                    const std::string& save_dir,
                     std::shared_ptr<pnnx::MainGraph>& pnnx_graph,
                     const std::string& device,
                     const std::vector<std::vector<int64_t> >& input_shapes,
                     const std::vector<std::string>& input_types,
                     const std::vector<std::vector<int64_t> >& input_shapes2,
                     const std::vector<std::string>& input_types2,
                     const std::vector<std::string>& customop_modules,
                     const std::vector<std::string>& module_operators,
                     const std::string& foldable_constants_zippath,
                     std::set<std::string>& foldable_constants);

} // namespace pnnx

#endif // PNNX_LOAD_TORCHSCRIPT_H
