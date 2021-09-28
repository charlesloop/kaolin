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

#include <stdlib.h>
#include <stdio.h>
#include <ATen/ATen.h>

#include "../../check.h"
#ifdef WITH_CUDA
#include "../../spc_math.h"
#endif

namespace kaolin {

// #define CHECK_TRIPLE(x) TORCH_CHECK(x.dim() == 2 && x.size(1) == 3, "input is not Nx3")
// #define CHECK_PACKED_FLOAT3(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x) CHECK_TRIPLE(x)
// #define CHECK_PACKED_LONG3(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_LONG(x) CHECK_TRIPLE(x)

using namespace std;
using namespace at::indexing;

#ifdef WITH_CUDA
// extern ulong GetStorageBytes(void* d_temp_storageA, morton_code* d_M0, morton_code* d_M1, uint max_total_points);

void SliceImage(uchar* octree, uchar* empty, uint level, uint* sum, uint offset, int axis, int voxel_slice, uint pixel_cnt, uchar* d_image);
void ClassifySpace(uchar* octree, uchar* empty, point_data* points, uint level, uint* pyramid, uint* sum);
#endif



at::Tensor classify_space(
    at::Tensor octree,
    at::Tensor points,
    uint level,
    at::Tensor pyramid,
    at::Tensor prefixsum)
{
  uint w = 0x1 << level;
  uint pixel_cnt = w*w;

  at::Tensor empty = at::zeros_like(octree);


  uchar* d_octree = octree.data_ptr<uchar>();
  uchar* d_empty = empty.data_ptr<uchar>();
  point_data* d_points = reinterpret_cast<point_data*>(points.data_ptr<short>());

  uint*  h_pyramid = reinterpret_cast<uint*>(pyramid.data_ptr<int>());
  uint*  d_sum = reinterpret_cast<uint*>(prefixsum.data_ptr<int>());


  ClassifySpace(d_octree, d_empty, d_points, level, h_pyramid, d_sum);

  return empty;
}




at::Tensor slice_image(
    at::Tensor octree,
    at::Tensor empty,
    at::Tensor points,
    uint level,
    at::Tensor pyramid,
    at::Tensor prefixsum,
    uint       axes,
    uint          val)
{
  uint w = 0x1 << level;
  uint pixel_cnt = w*w;

  at::Tensor Image = at::zeros({pixel_cnt}, octree.options().dtype(at::kByte));


  uchar* d_octree = octree.data_ptr<uchar>();
  uchar* d_empty = empty.data_ptr<uchar>();
  uint*  h_pyramid = reinterpret_cast<uint*>(pyramid.data_ptr<int>());
  uint*  d_sum = reinterpret_cast<uint*>(prefixsum.data_ptr<int>());
  uchar* d_image = Image.data_ptr<uchar>();

  auto pyramid_a = pyramid.accessor<int, 3>();
  uint offset = pyramid_a[0][1][level];

  SliceImage(d_octree, d_empty, level, d_sum, offset, axes, val, pixel_cnt, d_image);


  return Image;
}









}  // namespace kaolin
