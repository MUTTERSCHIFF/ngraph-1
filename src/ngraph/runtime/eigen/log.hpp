// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#pragma once

#include "ngraph/runtime/call_frame.hpp"
#include "ngraph/runtime/eigen/utils.hpp"
#include "ngraph/runtime/instruction.hpp"
#include "ngraph/runtime/tensor_view.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace eigen
        {
            template <typename ET>
            class LogInstruction : public Instruction
            {
            public:
                LogInstruction(TensorViewInfo arg, TensorViewInfo out)
                    : m_arg(arg)
                    , m_out(out)
                {
                }

                virtual void execute(CallFrame& call_frame) const override
                {
                    EigenArray1d<ET, fmt::V>(call_frame, m_out) = Eigen::log(EigenArray1d<ET, fmt::V>(call_frame, m_arg));
                }

            protected:
                TensorViewInfo m_arg;
                TensorViewInfo m_out;
            };
        }
    }
}
