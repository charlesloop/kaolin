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
#include "../../utils.h"

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



#define NUM_CHUNK 1536 //32*48
#define CHUNK 2097152 // 128^3
#define IMSIZE 25165824 // 4096 * 6144

__global__ void d_ImageMorton (
    const uint num, const uchar* image, uint bx, uint by, uchar* morder)
{
  uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

  if (tidx < num)
  {
    point_data p = ToPoint((morton_code)tidx);

    uchar val = 0;
    if (p.z < 120)
    {
      uint imidx = p.z*IMSIZE + (p.y+by)*6144 + p.x+bx;
      val = image[imidx];
    }

    morder[tidx] = val;
  }
}


void ImageStackToMorton_cuda(
  uchar* is, 
  uint depth, 
  uint height, 
  uint width, 
  uchar* mo,
  int* T)
{
  uchar* chunk = mo;
  for (int k = 0; k < 1536; k++)
  {
    morton_code m = T[k];
    point_data p = ToPoint(m);


    printf("%d   %d %d   %ld\n", k, p.x, p.y, m);


    d_ImageMorton <<< (CHUNK + 1023)/1024, 1024 >>>
    (
      CHUNK,
      is, 
      128*p.x,
      128*p.y,
      chunk);

    chunk += CHUNK;

    CUDA_CHECK(cudaGetLastError());

  }


  printf("done with cuda\n");



}




__global__ void d_FinalMip (
    const uint num, const uchar* morder, uchar2* mip)
{
  uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

  if (tidx < num)
  {
    uint midx = 8*tidx;
    uchar2 minmax = make_uchar2(255, 0);

    for (int i = 0; i < 8; i++)
    {
      uchar val = morder[midx + i];
      if (val < minmax.x) minmax.x = val;
      if (val > minmax.y) minmax.y = val;
    }

    mip[tidx] = minmax;
  }
}


__global__ void d_MiddleMip (
    const uint num, uchar2* mipin, uchar2* mipout)
{
  uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

  if (tidx < num)
  {
    uint midx = 8*tidx;
    uchar2 minmax = make_uchar2(255, 0);

    for (int i = 0; i < 8; i++)
    {
      uchar2 val = mipin[midx + i];
      if (val.x < minmax.x) minmax.x = val.x;
      if (val.y > minmax.y) minmax.y = val.y;
    }

    mipout[tidx] = minmax;    
  }
}



void BuildMip3D_cuda(uchar* mortonorder, uchar2* miplevels)
{
  uchar2* level_ptrs[7];
  uint64_t num_kernels[7];
  level_ptrs[0] = miplevels;
  num_kernels[0] = 1536;
  for (int l = 0; l < 6; l++)
  {
    level_ptrs[l+1] = level_ptrs[l] + 1536*(0x1<<(3*l));
    num_kernels[l+1] = 1536*(0x1<<(3*(l+1)));
    // printf("%d   %d  %d\n", l+1, 0x1<<(3*l), num_kernels[l+1]);
  }


  printf("launching %ld kernels\n", num_kernels[6]);
  d_FinalMip <<< (num_kernels[6] + 1023)/1024, 1024 >>> 
  (
    num_kernels[6],
    mortonorder, 
    level_ptrs[6]);

  CUDA_CHECK(cudaGetLastError());

  for (int l = 5; l >= 0; l--)
  {
    // printf("launching %ld kernels\n", num_kernels[l]);

    d_MiddleMip <<< (num_kernels[l] + 1023)/1024, 1024 >>> 
    (
      num_kernels[l],
      level_ptrs[l+1], 
      level_ptrs[l]);

    CUDA_CHECK(cudaGetLastError());

  }

}





ulong GetTempSize(void* d_temp_storage, uint* d_M0, uint* d_M1, uint max_total_points)
{
    ulong    temp_storage_bytes = 0;
    CubDebugExit(DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_M0, d_M1, max_total_points));
    return temp_storage_bytes;
}




__global__ void d_Decide (
    const uint num, point_data* points, uchar2* miplevels, uint val, uint* info)
{
  uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

  if (tidx < num)
  {
    morton_code m = ToMorton(points[tidx]);
    uchar2 bounds = miplevels[m];

    info[tidx] = bounds.y >= val-8 && bounds.x < val+8 ? 1 : 0;    
  }
}



void Decide_cuda(uint num, point_data* points, uchar2* miplevels, uint val, uint* info)
{

  d_Decide <<< (num + 1023)/1024, 1024 >>> 
  (
    num,
    points,
    miplevels,
    val,
    info);

  CUDA_CHECK(cudaGetLastError());


}






void InclusiveSum_cuda(uint num, uint* inputs, uint* outputs, void* d_temp_storage, ulong temp_storage_bytes)
{
  DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, inputs, outputs, num);
  CUDA_CHECK(cudaGetLastError());
}








__global__ void d_Compactify(uint numVoxels, point_data* voxelDataIn, point_data* voxelDataOut, uint* InSum)
{
	uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

	if (tidx < numVoxels)
	{
    uint IdxOut = (tidx == 0u) ? 0 : InSum[tidx-1];
		if (IdxOut != InSum[tidx])
		{
			voxelDataOut[IdxOut] = voxelDataIn[tidx];
		}
	}
}


void Compactify_cuda(uint num, point_data* points, uint* insum, point_data* new_points)
{
	if (num == 0u) return;

	d_Compactify << <(num + 1023) / 1024, 1024 >> >(num, points, new_points, insum);


}







__global__ void d_Subdivide(uint numVoxels, point_data* voxelDataIn, point_data* voxelDataOut, uint* InSum)
{
	uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

	if (tidx < numVoxels)
	{
    uint IdxOut = (tidx == 0u) ? 0 : InSum[tidx-1];
		if (IdxOut != InSum[tidx])
		{
			point_data Vin = voxelDataIn[tidx];
			point_data Vout = make_point_data(2 * Vin.x, 2 * Vin.y, 2 * Vin.z);

			uint IdxBase = 8 * IdxOut;

			for (uint i = 0; i < 8; i++)
			{
				voxelDataOut[IdxBase + i] = make_point_data(Vout.x + (i >> 2), Vout.y + ((i >> 1) & 0x1), Vout.z + (i & 0x1));
			}
		}
	}
}



void Subdivide_cuda(uint num, point_data* points, uint* exsum, point_data* new_points)
{
	if (num == 0u) return;

	d_Subdivide << <(num + 1023) / 1024, 1024 >> >(num, points, new_points, exsum);


}




__device__ float F(float x, float y, float z, float t)
{
  // return x*x + y*y + z*z -1.0f;

  return -1.0f/128.0f + (pow(x-t, 2) + y*y + z*z - 1.0f/64.0f)*(pow(x+t, 2) + y*y + z*z - 1.0f/64.0f);

}



__global__ void d_Contains(uint num, point_data* points, uint level, float t, uint* info)
{
	uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

	if (tidx < num)
	{
    point_data p = points[tidx];

    float minval = 1.0f;
    float maxval = -1.0f;

    float invL = 1.0f/(0x1<<(level-1));

    for (uint i = 0; i < 8; i++)
    {
      float x = invL*(p.x + (i >> 2)) - 1.0f;
      float y = invL*(p.y + ((i >> 1)&0x1)) - 1.0f;
      float z = invL*(p.z + (i & 0x1)) - 1.0f;
      float val = F(x, y, z, t);
      minval = fmin(minval, val);
      maxval = fmax(maxval, val);
    }

    info[tidx] = minval < 0.0 && maxval > 0.0 ? 1 : 0;    
	}
}




void Contains_cuda(uint num, point_data* points, uint level, float t, uint* info)
{

	if (num == 0u) return;

	d_Contains << <(num + 1023) / 1024, 1024 >> >(num, points, level, t, info);


}





__device__ float3 DF(float x, float y, float z, float t)
{
  float a = 4.0f*(x*x + y*y + z*z);
  float b = 4.0f*t*t;
  float c = 1.0f/16.0f;

  return make_float3(x*(a-b-c), y*(a+b-c), z*(a+b-c));

}



__global__ void d_CustomNormals(uint num, point_data* points, uint level, float t, float3* normals)
{
	uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

	if (tidx < num)
	{
    point_data p = points[tidx];

    float invL = 1.0f/(0x1<<(level-1));

    normals[tidx] = normalize(DF(invL*(p.x+0.5f) - 1.0f, invL*(p.y+0.5f) - 1.0f, invL*(p.z+0.5f) - 1.0f, t));

	}
}







void CustomNormals_cuda(uint num, point_data* points, uint level, float t, float3* normals)
{
	if (num == 0u) return;

	d_CustomNormals << <(num + 1023) / 1024, 1024 >> >(num, points, level, t, normals);


}





}  // namespace kaolin

