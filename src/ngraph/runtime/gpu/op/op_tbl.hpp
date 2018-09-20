//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#if defined(NGRAPH_OP_DISPATCH)
#define NGRAPH_OP(a) {type_index(typeid(ngraph::op::a)), runtime::gpu::GPU_Emitter::emit_##a},
#define NGRAPH_GPU_OP(a, NAMESPACE)                                                                \
    {type_index(typeid(NAMESPACE::a)), runtime::gpu::GPU_Emitter::emit_##a},
#elif defined(NGRAPH_OP_EMIT_DECL)
#define NGRAPH_OP(a) static void emit_##a(EMIT_ARGS);
#define NGRAPH_GPU_OP(a, NAMESPACE) static void emit_##a(EMIT_ARGS);
#endif

#include "ngraph/op/op_tbl.hpp"

#if CUDNN_VERSION >= 7200
NGRAPH_GPU_OP(Lstm, ngraph::op::gpu)
NGRAPH_GPU_OP(Rnn, ngraph::op::gpu)
#endif

#undef NGRAPH_OP
#undef NGRAPH_GPU_OP
