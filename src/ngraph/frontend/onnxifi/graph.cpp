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

#include <onnxifi.h>

#include "backend.hpp"
#include "graph.hpp"

namespace ngraph
{
    namespace onnxifi
    {
        bool Graph::run_graph()
        {
            std::cout << std::hex << "\n\tinput_event : 0x" << reinterpret_cast<uintptr_t>(m_input_fence->event)
                      << "\n\toutput_event: 0x" << reinterpret_cast<uintptr_t>(m_output_fence->event) << "\n";
            ::onnxStatus status{::onnxWaitEvent(m_input_fence->event)};
            if (status != ONNXIFI_STATUS_SUCCESS)
            {
                std::cout << "waitEvent(input): " << status::get_name(status) << "\n";
//                throw status::runtime{status};
            }
            bool result{m_backend.call(m_function, m_inputs, m_outputs)};
            status = ::onnxSignalEvent(m_output_fence->event);
            if (status != ONNXIFI_STATUS_SUCCESS)
            {
                std::cout << "signalEvent(output): " << status::get_name(status) << "\n";
                throw status::runtime{status};
            }
            return result;
        }

        void Graph::configure_memory_fences(const ::onnxMemoryFenceV1* input_fence,
                                            ::onnxMemoryFenceV1* output_fence)
        {
            if ((input_fence == nullptr) || (output_fence == nullptr))
            {
                throw status::null_pointer{};
            }
            if ((input_fence->tag != ONNXIFI_TAG_MEMORY_FENCE_V1) ||
                (output_fence->tag != ONNXIFI_TAG_MEMORY_FENCE_V1))
            {
                throw status::unsupported_tag{};
            }
            if ((input_fence->type == ONNXIFI_SYNCHRONIZATION_IMPLICIT) ||
                (output_fence->type == ONNXIFI_SYNCHRONIZATION_IMPLICIT))
            {
                throw status::unsupported_fence_type{};
            }
            if ((input_fence->type != ONNXIFI_SYNCHRONIZATION_EVENT) ||
                (output_fence->type != ONNXIFI_SYNCHRONIZATION_EVENT))
            {
                throw status::invalid_fence_type{};
            }
            ::onnxEventState state;
            ::onnxStatus status{::onnxGetEventState(output_fence->event, &state)};
            if (status == ONNXIFI_STATUS_INVALID_EVENT)
            {
                status = ::onnxInitEvent(m_backend.get_handle(), &output_fence->event);
                if (status != ONNXIFI_STATUS_SUCCESS)
                {
                    throw status::runtime{status};
                }
                status = ::onnxGetEventState(output_fence->event, &state);
            }
            if (status != ONNXIFI_STATUS_SUCCESS)
            {
                throw status::runtime{status};
            }
            if (state != ONNXIFI_EVENT_STATE_NONSIGNALLED)
            {
                throw status::invalid_state{};
            }
            status = ::onnxGetEventState(input_fence->event, &state);
            std::cout << "\n\t#1: input_fence : 0x" << std::hex << reinterpret_cast<uintptr_t>(input_fence->event)
                      << "\n\t    output_fence: 0x" << std::hex << reinterpret_cast<uintptr_t>(output_fence->event)
                      << "\n\t    status      : " << status::get_name(status) << "\n";
             if (status != ONNXIFI_STATUS_SUCCESS)
            {
                throw status::runtime{status};
            }
            m_input_fence = input_fence;
            m_output_fence = output_fence;
            std::cout << "\n\t#2: input_fence : 0x" << std::hex << reinterpret_cast<uintptr_t>(m_input_fence->event)
                      << "\n\t    output_fence: 0x" << std::hex << reinterpret_cast<uintptr_t>(m_output_fence->event) << "\n";
        }

        bool Graph::compile() { return m_backend.compile(m_function); }
        void Graph::load(std::istream& sin, const Span<::onnxTensorDescriptorV1>& weight_descriptors)
        {
            if (weight_descriptors.data() == nullptr)
            {
                throw status::null_pointer{};
            }
            if (weight_descriptors.empty())
            {
                throw status::invalid_size{};
            }
            onnx_import::Weights weights;
            for (const auto& weight : weight_descriptors)
            {
                InputTensor tensor{weight};
                auto get_type = [](onnxEnum dataType) -> onnx_import::Weight::Type {
                    switch (dataType)
                    {
                    case ONNXIFI_DATATYPE_FLOAT16:
                        return onnx_import::Weight::Type::f16;
                    case ONNXIFI_DATATYPE_FLOAT32:
                        return onnx_import::Weight::Type::f32;
                    case ONNXIFI_DATATYPE_FLOAT64:
                        return onnx_import::Weight::Type::f64;
                    case ONNXIFI_DATATYPE_INT8:
                        return onnx_import::Weight::Type::i8;
                    case ONNXIFI_DATATYPE_INT16:
                        return onnx_import::Weight::Type::i16;
                    case ONNXIFI_DATATYPE_INT32:
                        return onnx_import::Weight::Type::i32;
                    case ONNXIFI_DATATYPE_INT64:
                        return onnx_import::Weight::Type::i64;
                    case ONNXIFI_DATATYPE_UINT8:
                        return onnx_import::Weight::Type::u8;
                    case ONNXIFI_DATATYPE_UINT16:
                        return onnx_import::Weight::Type::u16;
                    case ONNXIFI_DATATYPE_UINT32:
                        return onnx_import::Weight::Type::u32;
                    case ONNXIFI_DATATYPE_UINT64:
                        return onnx_import::Weight::Type::u64;
                    default:
                        throw status::invalid_datatype{};
                    }
                };

                weights.emplace_back(weight.name, get_type(weight.dataType),
                     weight.dimensions, weight.shape, reinterpret_cast<const void*>(weight.buffer));
            }
            m_function = onnx_import::import_onnx_function(sin, weights);
        }

    } // namespace onnxifi

} // namespace ngraph
