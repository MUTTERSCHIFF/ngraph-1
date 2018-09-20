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

#include <cmath>
#include <iostream>
#include <vector>
#include <iostream>

#include "ngraph/axis_vector.hpp"
#include "ngraph/coordinate_transform.hpp"
#include "ngraph/runtime/reference/add.hpp"
#include "ngraph/runtime/reference/broadcast.hpp"
#include "ngraph/runtime/reference/divide.hpp"
#include "ngraph/runtime/reference/multiply.hpp"
#include "ngraph/runtime/reference/sum.hpp"
#include "ngraph/util.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename TI, typename TO>
            void quantize(
                const TI* input, 
                const TI* scale, 
                const TI* offset, 
                TO* output, 
                const Shape& input_shape, 
                const Shape& scale_offset_shape,
                const AxisSet& axes)
            {
                CoordinateTransform input_transform(input_shape);
                CoordinateTransform scale_offset_transform(scale_offset_shape);

                for (const Coordinate& input_coord : input_transform)
                {
                    Coordinate scale_offset_coord;
                    // TODO: project_for_realz
                    for (size_t i = 0; i < input_coord.size(); ++i)
                    {
                        if (axes.find(i) != axes.end())
                        {
                            scale_offset_coord.push_back(input_coord[i]);
                        }
                    }

                    output[input_transform.index(input_coord)] = 
                        static_cast<TO>(
                            std::round(
                                input[input_transform.index(input_coord)] / 
                                scale[scale_offset_transform.index(scale_offset_coord)]
                            )
                            + offset[scale_offset_transform.index(scale_offset_coord)]
                        );
                }
            }
        }
    }
}
