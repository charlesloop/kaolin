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

#include <iostream>

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
void ImageStackToMorton_cuda(
  uchar* is, 
  uint depth, 
  uint height, 
  uint width, 
  uchar* mo,
  int* ofs);
void BuildMip3D_cuda(uchar* mortonorder, uchar2* miplevels);
void Decide_cuda(uint num, point_data* points, uchar2* miplevels, uint val, uint* info);

ulong GetTempSize(void* d_temp_storage, uint* d_M0, uint* d_M1, uint max_total_points);
void InclusiveSum_cuda(uint num, uint* inputs, uint* outputs, void* d_temp_storage, ulong temp_storage_bytes);
void Compactify_cuda(uint num, point_data* points, uint* insum, point_data* new_points);
void Subdivide_cuda(uint num, point_data* points, uint* insum, point_data* new_points);
void Contains_cuda(uint num, point_data* points, uint level, float t, uint* info);
void CustomNormals_cuda(uint num, point_data* points, uint level, float t, float3* normals);
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




#define CHUNK 2097152

at::Tensor imagestack_to_morton(
    at::Tensor imagestack,
    at::Tensor offsets)
{
  uint depth = imagestack.size(0);
  uint height = 4096;//imagestack.size(1);
  uint width = 6144;//imagestack.size(2);

  // std::cout << imagestack.options() << "\n";
  // printf("%u  %u    %u\n", 32*48*CHUNK , depth*height*width, 0xFFFFFFFF );

  at::Tensor mortonorder = at::zeros({ 32*48, CHUNK }, imagestack.options()); // trim excess later

  uchar* is = imagestack.data_ptr<uchar>();
  uchar* mo = mortonorder.data_ptr<uchar>();
  int* ofs = offsets.data_ptr<int>();

  ImageStackToMorton_cuda(is, depth, height, width, mo, ofs);


  return mortonorder;
}




at::Tensor build_mip3d(at::Tensor mortonorder)
{

 
  uint size = 0;
  for (int l = 0; l < 7; l++)
    size += 0x1<<(3*l);

  // printf("before\n");

  at::Tensor miplevels = at::empty({ 1536*size, 2 }, mortonorder.options());

  uchar* mo = mortonorder.data_ptr<uchar>();
  uchar2* mip = reinterpret_cast<uchar2*>(miplevels.data_ptr<uchar>());

  BuildMip3D_cuda(mo, mip);

  // printf("after\n");

  return miplevels;

}



at::Tensor decide(at::Tensor Points, at::Tensor miplevels, uint level, uint val)
{
  uint num = Points.size(0);

  // printf("level = %d   num = %d\n", level, num);

  uchar2* mip = reinterpret_cast<uchar2*>(miplevels.data_ptr<uchar>()); 
  uchar2* level_ptrs[7];
  level_ptrs[0] = mip;
  for (int l = 0; l < 6; l++)
  {
    level_ptrs[l+1] = level_ptrs[l] + 1536*(0x1<<(3*l));
  }

  at::Tensor occupancy = at::ones({num}, miplevels.options().dtype(at::kInt));
  uint* occ = reinterpret_cast<uint*>(occupancy.data_ptr<int>());

  point_data*  points = (point_data*)Points.data_ptr<short>();

  Decide_cuda(num, points, level_ptrs[level], val, occ);

//  if (level < 1)
//   {
//     TORCH_CHECK(num == 1536 && level == 0, "bad inpus to contains");
//     // occupancy[0] = 1;
//   }
//   else
//   {
//     Decide_cuda(num, points, level_ptrs[level], val, occ);
//   }

  return occupancy;
}




at::Tensor inclusive_sum(at::Tensor Inputs)
{
  uint num = Inputs.size(0);

  at::Tensor Outputs = at::zeros_like(Inputs);

  uint* inputs = reinterpret_cast<uint*>(Inputs.data_ptr<int>()); 
  uint* outputs = reinterpret_cast<uint*>(Outputs.data_ptr<int>());

  // set up memory for DeviceScan and DeviceRadixSort calls
  void* d_temp_storage = NULL;
  ulong temp_storage_bytes = GetTempSize(d_temp_storage, inputs, outputs, num);

  at::Tensor temp_storage = at::zeros({(long)temp_storage_bytes}, Inputs.options().dtype(at::kByte));
  d_temp_storage = (void*)temp_storage.data_ptr<uchar>();

  InclusiveSum_cuda(num, inputs, outputs, d_temp_storage, temp_storage_bytes);

  return Outputs;
}


at::Tensor compactify(at::Tensor Points, at::Tensor Insum)
{
  uint num = Points.size(0);
  uint pass = Insum[-1].item<int>();

  at::Tensor NewPoints = at::zeros({pass, 3}, Points.options());

  point_data*  points = (point_data*)Points.data_ptr<short>();
  point_data*  new_points = (point_data*)NewPoints.data_ptr<short>();

  uint* insum = reinterpret_cast<uint*>(Insum.data_ptr<int>());

  Compactify_cuda(num, points, insum, new_points);

  return NewPoints;
}



at::Tensor subdivide(at::Tensor Points, at::Tensor Insum)
{
  uint num = Points.size(0);
  uint pass = Insum[-1].item<int>();

  at::Tensor NewPoints = at::zeros({8*pass, 3}, Points.options());

  point_data*  points = (point_data*)Points.data_ptr<short>();
  point_data*  new_points = (point_data*)NewPoints.data_ptr<short>();

  uint* insum = reinterpret_cast<uint*>(Insum.data_ptr<int>());

  Subdivide_cuda(num, points, insum, new_points);

  return NewPoints;
}


at::Tensor contains(at::Tensor Points, uint level, float t)
{
  uint num = Points.size(0);

  // printf("level = %d   num = %d\n", level, num);

  at::Tensor occupancy = at::zeros({num}, Points.options().dtype(at::kInt));

  uint* occ = reinterpret_cast<uint*>(occupancy.data_ptr<int>());
  point_data*  points = (point_data*)Points.data_ptr<short>();

  if (level < 1)
  {
    TORCH_CHECK(num == 1 && level == 0, "bad inpus to contains");
    occupancy[0] = 1;
  }
  else
  {
    Contains_cuda(num, points, level, t, occ);
  }

  return occupancy;

}



at::Tensor custom_normals(at::Tensor Points, uint level, float t)
{
  uint num = Points.size(0);

  at::Tensor Normals = at::zeros({num, 3}, Points.options().dtype(at::kFloat));

  point_data*  points = (point_data*)Points.data_ptr<short>();
  float3* normals = reinterpret_cast<float3*>(Normals.data_ptr<float>());

  CustomNormals_cuda(num, points, level, t, normals);

  return Normals;
}

}  // namespace kaolin
