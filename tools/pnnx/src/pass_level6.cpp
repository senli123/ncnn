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

#include "pass_level6.h"

#include "pass_level6/eliminate_ListUnpack.h"
#include "pass_level6/trans_expression2TupleConstruct.h"
#include "pass_level6/trans_Stack2Unsqueeze.h"
#include "pass_level6/trans_ReshapeAs2Reshape.h"
#include "pass_level6/trans_TensorTypeAs2TensorTo.h"

namespace pnnx {

void pass_level6(Graph& g, const std::set<std::string>& foldable_constants, const std::string& foldable_constants_zippath)
{
    eliminate_ListUnpack(g);
    fprintf(stderr, "############# finish eliminate_ListUnpack\n");
    trans_expression2TupleConstruct(g);
    fprintf(stderr, "############# finish trans_expression2TupleConstruct\n");
    trans_Stack2Unsqueeze(g);
    fprintf(stderr, "############# finish trans_Stack2Unsqueeze\n");
    trans_ReshapeAs2Reshape(g);
    fprintf(stderr, "############# finish trans_ReshapeAs2Reshape\n");
    trans_TensorTypeAs2TensorTo(g);
    fprintf(stderr, "############# finish trans_TensorTypeAs2TensorTo\n");
}

} // namespace pnnx
