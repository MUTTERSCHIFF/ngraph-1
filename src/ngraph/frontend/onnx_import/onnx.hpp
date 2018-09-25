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

#pragma once

#include <iostream>
#include <map>
#include <string>

#include "ngraph/function.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        class Weight
        {
        public:
            enum class Type
            {
                f16, f32, f64,
                i8, i16, i32, i64,
                u8, u16, u32, u64
            };

            Weight(const char* name, Type type, std::size_t dimensions, const std::size_t* shape, const void* buffer)
                : m_name{name},
                  m_type{type},
                  m_dimensions{dimensions},
                  m_shape{shape},
                  m_buffer{buffer}
            {
            }

            Weight() = delete;

            Weight(const Weight&) = default;
            Weight& operator=(const Weight&) = default;

            Weight(Weight&&) noexcept = default;
            Weight& operator=(Weight&&) = default;

            const char* name() const
            {
                return m_name;
            }

            const std::size_t* shape() const
            {
                return m_shape;
            }

            Type type() const
            {
                return m_type;
            }

            std::size_t dimensions() const
            {
                return m_dimensions;
            }

            const void* data() const
            {
                return m_buffer;
            }

        private:
            const char* m_name;
            Type m_type;
            std::size_t m_dimensions;
            const std::size_t* m_shape;
            const void* m_buffer;
        };

        using Weights = std::vector<Weight>;

        // Convert on ONNX model to a vector of nGraph Functions (input stream)
        std::vector<std::shared_ptr<Function>> load_onnx_model(std::istream&, const Weights& weights = {});

        // Convert an ONNX model to a vector of nGraph Functions
        std::vector<std::shared_ptr<Function>> load_onnx_model(const std::string&, const Weights& weights = {});

        // Convert the first output of an ONNX model to an nGraph Function (input stream)
        std::shared_ptr<Function> import_onnx_function(std::istream&, const Weights& weights = {});

        // Convert the first output of an ONNX model to an nGraph Function
        std::shared_ptr<Function> import_onnx_function(const std::string&, const Weights& weights = {});

    } // namespace onnx_import

} // namespace ngraph
