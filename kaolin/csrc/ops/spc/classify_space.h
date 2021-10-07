// Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//    http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef KAOLIN_OPS_CONVERSIONS_SPC_CLASSIFY_SPACE_H_
#define KAOLIN_OPS_CONVERSIONS_SPC_CLASSIFY_SPACE_H_

#include <ATen/ATen.h>



namespace kaolin {

at::Tensor classify_space(
    at::Tensor octree,
    at::Tensor points,
    uint level,
    at::Tensor pyramid,
    at::Tensor prefixsum);


at::Tensor slice_image(
    at::Tensor octree,
    at::Tensor empty,
    at::Tensor points,
    uint level,
    at::Tensor pyramid,
    at::Tensor prefixsum,
    uint axes,
    uint val);



at::Tensor imagestack_to_morton(
    at::Tensor imagestack,
    at::Tensor offsets);

at::Tensor build_mip3d(at::Tensor volumedata);

at::Tensor decide(at::Tensor Points, at::Tensor miplevels, uint level, uint val);
at::Tensor inclusive_sum(at::Tensor Inputs);
at::Tensor compactify(at::Tensor Points, at::Tensor Exsum);
at::Tensor subdivide(at::Tensor Points, at::Tensor Exsum);
at::Tensor contains(at::Tensor Points, uint level, float t);
at::Tensor custom_normals(at::Tensor Points, uint level, float t);

}  // namespace kaolin

#endif  // KAOLIN_OPS_CONVERSIONS_SPC_CLASSIFY_SPACE_H_
