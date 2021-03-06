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

#include <onnx.pb.h>
#include <string>
#include <vector>

#include "ngraph/op/parameter_vector.hpp"

#include "value_info.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        class Graph
        {
        public:
            explicit Graph(const onnx::GraphProto& proto);

            const std::vector<Node>& get_nodes() const { return m_nodes; }
            const std::vector<ValueInfo>& get_inputs() const { return m_inputs; }
            const std::vector<ValueInfo>& get_outputs() const { return m_outputs; }
            const op::ParameterVector& get_ng_parameters() const { return m_parameters; }
            std::shared_ptr<ngraph::Node> get_ng_node_from_cache(const std::string& name) const
            {
                return m_ng_node_cache.at(name);
            }

            const std::string& get_name() const { return m_graph_proto->name(); }
        private:
            const onnx::GraphProto* m_graph_proto;
            std::vector<Node> m_nodes;
            std::vector<ValueInfo> m_inputs;
            std::vector<ValueInfo> m_outputs;
            op::ParameterVector m_parameters;
            std::map<std::string, std::shared_ptr<ngraph::Node>> m_ng_node_cache;
            std::map<std::string, Tensor> m_initializers;
        };

        inline std::ostream& operator<<(std::ostream& outs, const Graph& graph)
        {
            return (outs << "<Graph: " << graph.get_name() << ">");
        }

    } // namespace onnx_import

} // namespace ngraph
