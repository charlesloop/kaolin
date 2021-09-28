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

#define CUB_NS_PREFIX namespace kaolin {
#define CUB_NS_POSTFIX }

#include <torch/extension.h>
#include <stdio.h>

#define CUB_STDERR
#include <cub/device/device_scan.cuh>
#include <cub/device/device_radix_sort.cuh>

#include "../../spc_math.h"

namespace kaolin {

using namespace cub;
using namespace std;



/////////// empty-unseen //////////////////////////


__device__ int Identify(
  const short3 	k,
  const uint 		Level,
  const uint* 	Exsum,
  const uchar* 	Oroot,
  uchar* 	Eroot,
  const uint 		offset)
{
  int maxval = (0x1 << Level) - 1; // seems you could do this better using Morton codes
  if (k.x < 0 || k.y < 0 || k.z < 0 || k.x > maxval || k.y > maxval || k.z > maxval)
    return -1;

  int ord = 0;
  int prev = 0;
  for (uint l = 0; l < Level; l++)
  {
    uint depth = Level - l - 1;
    uint mask = (0x1 << depth);
    uint child_idx = ((mask&k.x) << 2 | (mask&k.y) << 1 | (mask&k.z)) >> depth;
    uint bits = (uint)Oroot[ord];
    uint mpty = (uint)Eroot[ord];

    // count set bits up to child - inclusive sum
    uint cnt = __popc(bits&((0x2 << child_idx) - 1));
    ord = Exsum[prev];

    // if bit set, keep going
    if (bits&(0x1 << child_idx))
    {
      ord += cnt;

      if (depth == 0)
        return ord - offset;
    }
    else
    {
      if (mpty&(0x1 << child_idx))
        return -2 - depth;
      else
        return -1;
    }

    prev = ord;
  }

  return ord; // only if called with Level=0
}


__global__ void d_ClassifySpace (
  const uint 	num_nodes, 
  const uint offset,
  const uchar* 	octree, 
  const point_data* 	points, 
  const uint 	level,
  const uint* 	exsum,  
  uchar* 	empty)
{
  uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

  if (tidx < num_nodes)
  {
    uint node = (uint)octree[offset + tidx];
    short3 p = points[offset + tidx];
    
    uint eout = 255;
    int nx[2], ny[2], nz[2];

    short3 v;    
    for (int i = 0; i < 2; i++)
    {
      nx[i] = Identify(make_short3(p.x+2*i-1, p.y, p.z), level, exsum, octree, empty, offset);
      ny[i] = Identify(make_short3(p.x, p.y+2*i-1, p.z), level, exsum, octree, empty, offset);
      nz[i] = Identify(make_short3(p.x, p.y, p.z+2*i-1), level, exsum, octree, empty, offset);
    }

 
    for (int k = 0; k < 8; k++)
    {
      int x = (k >> 2) & 1;
      int y = (k >> 1) & 1;
      int z = k & 1;

      int xb = 1 - x;
      int yb = 1 - y;
      int zb = 1 - z;

      int xh = 4*xb + 2*y + z;
      int yh = 4*x + 2*yb + z;
      int zh = 4*x + 2*y + zb;

      if (node&(0x1<<k))
      {
        eout ^= (0x1<<k);
      }
      else // empty cell
      {

        if (
          nz[z] == -1 || 
          ny[y] == -1 || 
          nx[x] == -1 || 
          (!(node&(1<<xh)) && nx[xb] == -1) || 
          (!(node&(1<<yh)) && ny[yb] == -1) || 
          (!(node&(1<<zh)) && nz[zb] == -1) )
        {
          eout ^= (0x1<<k);
        }
      }
    }

    uint ein = eout;

    for (int k = 0; k < 8; k++)
    {
      int x = (k >> 2) & 1;
      int y = (k >> 1) & 1;
      int z = k & 1;

      int xb = 1 - x;
      int yb = 1 - y;
      int zb = 1 - z;

      int xh = 4*xb + 2*y + z;
      int yh = 4*x + 2*yb + z;
      int zh = 4*x + 2*y + zb;

      if (eout&(0x1<<k))
      {
        if ( (!(node&(1<<xh)) && !(ein&(1<<xh))) ||
             (!(node&(1<<yh)) && !(ein&(1<<yh))) ||
             (!(node&(1<<zh)) && !(ein&(1<<zh))) )
        {
          eout ^= (0x1<<k);     
        }
      }
    }

    empty[offset + tidx] = (uchar)eout;

  }
}


__global__ void d_ConsistencyPass (
  const uint 	num_nodes, 
  const uint offset,
  const uchar* 	octree, 
  const point_data* 	points, 
  const uint 	level,
  const uint* 	exsum,  
  uchar* 	empty)
{
  uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

  if (tidx < num_nodes)
  {
    uint node = (uint)octree[offset + tidx];
    uint eout = (uint)empty[offset + tidx];
    short3 p = points[offset + tidx];

    for (int k = 0; k < 8; k++)
    {
      int x = (k >> 2) & 1;
      int y = (k >> 1) & 1;
      int z = k & 1;

      short3 q = make_short3(2*p.x+x, 2*p.y+y, 2*p.z+z);

      int nx = Identify(make_short3(q.x+2*x-1, q.y, q.z), level+1, exsum, octree, empty, offset+num_nodes);
      int ny = Identify(make_short3(q.x, q.y+2*y-1, q.z), level+1, exsum, octree, empty, offset+num_nodes);
      int nz = Identify(make_short3(q.x, q.y, q.z+2*z-1), level+1, exsum, octree, empty, offset+num_nodes);

      if (eout&(0x1<<k))// unseen cell
      {
        if (
          nz == -1 || 
          ny == -1 || 
          nx == -1 )
        {
          eout ^= (0x1<<k);
        }
      }
    }

    empty[offset + tidx] = (uchar)eout;

  }
}


void ClassifySpace(uchar* octree, uchar* empty, point_data* points, uint level, uint* pyramid, uint* sum)
{
  uint ord, num_nodes = 0;
  for (int i = 0; i < level; i++)
  {
    ord = pyramid[i];

    printf("%d  %d %d\n", i, ord, num_nodes);

    d_ClassifySpace << <(ord + 63) / 64, 64 >> > 
    (
      ord, 
      num_nodes,
      octree,
      points,
      i,
      sum,
      empty);   

    d_ConsistencyPass << <(ord + 63) / 64, 64 >> > 
    (
      ord, 
      num_nodes,
      octree,
      points,
      i,
      sum,
      empty);   

    num_nodes += ord;
  }

  printf("%d  %d %d\n", level, pyramid[level], num_nodes);

}


__global__ void d_SliceImage (
  const uint 	pixel_cnt, 
  const int axes,
  const int	  voxel_slice,
  const uint* 	exsum, 
  const uchar* 	octree, 
  uchar* 	empty, 
  const uint 	level,
  const uint 	offset,			
  uchar* 		image)
{
  uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

  if (tidx < pixel_cnt)
  {
    short3 W;

    int res = 0x1 << level;

    switch (axes)
    {
      case 0:
        W.z = tidx / res;
        W.x = voxel_slice;
        W.y = tidx % res;
        break;
      case 1:
        W.x = tidx / res;
        W.y = voxel_slice;
        W.z = tidx % res;
        break;
      case 2:
        W.y = tidx / res;
        W.z = voxel_slice;
        W.x = tidx % res;
        break;
    }

    int id = Identify(W, level, exsum, octree, empty, offset);

    if (W.x == 0 && W.y == 0 && W.z == 0) printf("%d  %d   %d\n", tidx, id, offset);

    uchar clr;

    if (id < 0)
      clr = id == -1 ? 0 : 64+16*(id+9);
    else
      clr = 255;

    image[tidx] = clr;

  }
}


void SliceImage(uchar* octree, uchar* empty, uint level, uint* sum, uint offset, int axes, int voxel_slice, uint pixel_cnt, uchar* d_image)
{
  d_SliceImage << <(pixel_cnt + 63) / 64, 64 >> > 
  (
    pixel_cnt, 
    axes,
    voxel_slice,
    sum,
    octree,
    empty,
    level,
    offset,
    d_image);
}







}  // namespace kaolin

