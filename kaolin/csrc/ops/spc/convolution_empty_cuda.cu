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


#include "../../utils.h"
#include "convolution.cuh"

#define CUB_NS_PREFIX namespace kaolin {
#define CUB_NS_POSTFIX }

#include <cub/device/device_scan.cuh>

namespace kaolin {

using namespace cub;

#define THREADS_PER_BLOCK 64

namespace minkowski {

  template <typename Dtype, typename Itype>
  void ConvolutionForwardKernelGPU(const Dtype *d_in_feat, int in_nchannel,
      Dtype *d_out_feat, int out_nchannel,
      const Dtype *d_kernel,
      const pInOutMaps<Itype> &in_map,
      const pInOutMaps<Itype> &out_map,
      int out_nrows, cublasHandle_t cuhandle,
      cudaStream_t stream);

  template <typename Dtype, typename Itype>
  void ConvolutionBackwardKernelGPU(const Dtype *d_in_feat, Dtype *d_grad_in_feat,
      int in_nchannel, const Dtype *d_grad_out_feat,
      int out_nchannel, const Dtype *d_kernel,
      Dtype *d_grad_kernel,
      const pInOutMaps<Itype> &in_map,
      const pInOutMaps<Itype> &out_map,
      int out_nrows, cublasHandle_t cuhandle,
      cudaStream_t stream);

} //end namespace minkowski


const char *cublasGetErrorString(cublasStatus_t error) {
  switch (error) {
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";
  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";
  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";
  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";
  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";
  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";
  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";
  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";
#if CUDA_VERSION >= 6000
  case CUBLAS_STATUS_NOT_SUPPORTED:
    return "CUBLAS_STATUS_NOT_SUPPORTED";
#endif
#if CUDA_VERSION >= 6050
  case CUBLAS_STATUS_LICENSE_ERROR:
    return "CUBLAS_STATUS_LICENSE_ERROR";
#endif
  }
  return "Unknown cublas status";
}


#define CUBLAS_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cublasStatus_t status = condition; \
    CHECK_EQ(status, CUBLAS_STATUS_SUCCESS) << " " << cublasGetErrorString(status); \
  } while (0)


uint GetPyramid(uint* Pyramid, int batch, int k, int level, int olevel) ;

long GetStorageBytes(void* d_temp_storage, uint* d_Info,
                       uint* d_Exsum, uint max_total_points);


__device__ int Identify(
  const point_data 	k,
  const uint 		Level,
  const uint* 		Exsum,
  const uchar* 		Oroot,
  const uchar* 		Eroot,
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
        return -2;
      else
        return -1;
    }

    prev = ord;
  }

  return ord; // only if called with Level=0
}


__global__ void GenerateKernelMap(
    const uint num,
    const point_data* Pdata,
    int*  Inmap,
    int*  Outmap,
    uint*  Info,
    const uint K, 
    const point_data* Kvec,
    const int scale,
    const uchar* Oroot, 
    const uchar* Eroot, 
    const uint* Exsum,
    const uint level, 
    const uint offset,
    float* Empty,
    float* Unseen)
{
  int o_idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (o_idx < num)
  {
    point_data V = mul_point_data(scale, Pdata[o_idx]);
    Outmap[o_idx] = o_idx;

    for (int k = 0; k < K; k++)
    {
      int i_idx = Identify(add_point_data(V, Kvec[k]), level, Exsum, Oroot, Eroot, offset);

      int adr = k*num + o_idx;
      int adr1 = K*o_idx + k;
      Inmap[adr] = i_idx;
      if (i_idx >= 0)
      {
        Info[adr] = 1;
        Empty[adr1] = 0.0f;
        Unseen[adr1] = 0.0f;
      }
      else if (i_idx == -1)
      {
        Info[adr] = 0;
        Empty[adr1] = 1.0f;
        Unseen[adr1] = 0.0f;
      }
      else
      {
        Info[adr] = 0;
        Empty[adr1] = 0.0f;
        Unseen[adr1] = 1.0f;
      }
     }
  }
}

__global__ void GenerateKernelMap2(
    const uint num,
    const point_data* Pdata,
    const uint K, 
    const point_data* Kvec,
    const int scale,
    const uchar* Oroot, 
    const uchar* Eroot, 
    const uint* Exsum,
    const uint level, 
    const uint offset,
    float* Empty,
    float* Unseen)
{
  int o_idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (o_idx < num)
  {
    point_data V = mul_point_data(scale, Pdata[o_idx]);

    for (int k = 0; k < K; k++)
    {
      int i_idx = Identify(add_point_data(V, Kvec[k]), level, Exsum, Oroot, Eroot, offset);

      int adr1 = K*o_idx + k;
      if (i_idx >= 0)
      {
        Empty[adr1] = 0.0f;
        Unseen[adr1] = 0.0f;
      }
      else if (i_idx == -1)
      {
        Empty[adr1] = 1.0f;
        Unseen[adr1] = 0.0f;
      }
      else
      {
        Empty[adr1] = 0.0f;
        Unseen[adr1] = 1.0f;
      }
     }
  }
}


__global__ void GenerateKernelMapTrans(
    const uint num,
    const point_data* Pdata,
    int*  Inmap,
    int*  Outmap,
    uint*  Info,
    const uint K, 
    const point_data* Kvec,
    const int scale,
    uchar* Oroot, 
    const uchar* Eroot, 
    uint* Exsum,
    uint level, 
    uint offset,
    float* Empty,
    float* Unseen)
{
  int o_idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (o_idx < num) {
    point_data V = Pdata[o_idx];
    Outmap[o_idx] = o_idx;

    for (int k = 0; k < K; k++) {
      point_data U = sub_point_data(V, Kvec[k]);
      int adr = k*num + o_idx;
      int adr1 = K*o_idx + k;

      if (U.x%scale == 0 && U.y%scale == 0 && U.z%scale == 0) {
        int i_idx = Identify(div_point_data(U, scale), level, Exsum, Oroot, Eroot, offset);

        Inmap[adr] = i_idx;
        if (i_idx >= 0)
        {
            Info[adr] = 1;
            Empty[adr1] = 0.0f;
            Unseen[adr1] = 0.0f;
        }
        else if (i_idx == -1)
        {
            Info[adr] = 0;
            Empty[adr1] = 1.0f;
            Unseen[adr1] = 0.0f;
        }
        else
        {
            Info[adr] = 0;
            Empty[adr1] = 0.0f;
            Unseen[adr1] = 1.0f;
        }      
      } else {
        Info[adr] = 0;
        Empty[adr1] = 0.0f;
        Unseen[adr1] = 0.0f;
      }
    }
  }
}


__global__ void GenerateKernelMapTrans2(
    const uint num,
    const point_data* Pdata,
    const uint K, 
    const point_data* Kvec,
    const int scale,
    uchar* Oroot, 
    const uchar* Eroot, 
    uint* Exsum,
    uint level, 
    uint offset,
    float* Empty,
    float* Unseen)
{
  int o_idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (o_idx < num) {
    point_data V = Pdata[o_idx];

    for (int k = 0; k < K; k++) {
      point_data U = sub_point_data(V, Kvec[k]);
      int adr1 = K*o_idx + k;

      if (U.x%scale == 0 && U.y%scale == 0 && U.z%scale == 0) {
        int i_idx = Identify(div_point_data(U, scale), level, Exsum, Oroot, Eroot, offset);

        if (i_idx >= 0)
        {
            Empty[adr1] = 0.0f;
            Unseen[adr1] = 0.0f;
        }
        else if (i_idx == -1)
        {
           Empty[adr1] = 1.0f;
            Unseen[adr1] = 0.0f;
        }
        else
        {
            Empty[adr1] = 0.0f;
            Unseen[adr1] = 1.0f;
        }      
      } else {
        Empty[adr1] = 0.0f;
        Unseen[adr1] = 0.0f;
      }
    }
  }
}

__global__ void CompactifyMaps(
    const uint OutSize,
    const uint num,
    const int *Inmap,
    const int *Outmap,
    int *InmapX,
    int *OutmapX,
    const uint *Info,
    const uint *Exsum) ;


void ProcessKernelMaps(
    uint K,
    uint Cnt,
    pInOutMaps<int32_t> &in_map,
    pInOutMaps<int32_t> &out_map,
    uint* Info,
    uint* PSum,
    void* d_temp_storageA,
    size_t temp_storage_bytesA,
    int* Inmap,
    int* Outmap,
    int* InmapX,
    int* OutmapX) ;


void UnoccupiedContribution(
    cublasHandle_t handle,
    int OutSize, int Ksize, int Cin, int Cout,
    float*      W,
    float*      Indicator,
    float*      Q,
    float*      C,
    float*      single_input)
{
    float alpha = 1.0f;
    float beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(handle, 
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             Cout*Cin, OutSize, Ksize, 
                             &alpha, 
                             W, Cout*Cin,          // B
                             Indicator, Ksize,        // A
                             &beta, 
                             Q, Cout*Cin));    // C
    beta = 1.0;



    // CUBLAS_CHECK(cublasSgemm(handle, 
    //                          CUBLAS_OP_N, CUBLAS_OP_N,
    //                          1, OutSize*Cout, Cin, 
    //                          &alpha, 
    //                          single_input, 1,          // B
    //                          Q, Cin,        // A
    //                          &beta, 
    //                          C, 1));    // C

    CUBLAS_CHECK(cublasSgemm(handle, 
                             CUBLAS_OP_T, CUBLAS_OP_N,
                             1, OutSize*Cout, Cin, 
                             &alpha, 
                             single_input, Cin,          // B
                             Q, Cin,        // A
                             &beta, 
                             C, 1));    // C
}






void UnoccupiedWeightUpdate(
  cublasHandle_t handle,
  int OutSize, int Ksize, int Cout, int Cin,
  float*      dW,
  float*      Indicator,
  float*      H,
  float*      dY,
  float*      A)
  {
    float alpha = 1.0f;
    float beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(handle, 
                             CUBLAS_OP_N, CUBLAS_OP_T,
                             Ksize, Cout, OutSize, 
                             &alpha, 
                             Indicator, Ksize,     // A
                             dY, Cout,        // B
                             &beta, 
                             H, Ksize));   // C


    float* C = dW;
    float* B = H; 


    // float* hA = new float[Cin];
    // float* hB = new float[Cout];
    // float* hC = new float[Cin*Cout];

    // cudaMemcpy(hA, A, Cin*sizeof(float), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < Cin; i++)
    //   printf("%f ", hA[i]);
    // printf("\n");

    beta = 1.0;

    for (int k = 0; k < Ksize; k++)
    {
      // cudaMemcpy(hB, B, Cout*sizeof(float), cudaMemcpyDeviceToHost);
      // for (int i = 0; i < Cout; i++)
      //   printf("%f ", hB[i]);
      // printf("\n");


      // CUBLAS_CHECK(cublasSgemm(handle, 
      //   CUBLAS_OP_T, CUBLAS_OP_N,
      //   Cout, Cin, 1, 
      //   &alpha, 
      //   B, 1,     // A
      //   A, 1,        // B
      //   &beta, 
      //   C, Cout));   // C

      CUBLAS_CHECK(cublasSgemm(handle, 
        CUBLAS_OP_N, CUBLAS_OP_T,
        Cout, Cin, 1, 
        &alpha, 
        B, Cout,     // A
        A, Cin,        // B
        &beta, 
        C, Cout));   // C

      // cudaMemcpy(hC, C, Cout*Cin*sizeof(float), cudaMemcpyDeviceToHost);
      // for (int i = 0; i < Cout*Cin; i++)
      //   printf("%f ", hC[i]);
      // printf("\n\n");



      B += Cout;
      C += Cout*Cin;
    }
    // printf("\n");
 
    // printf("\n------------------\n");
    
    // C = dW;
    // cudaMemcpy(hC, C, Ksize*Cin*Cout*sizeof(float), cudaMemcpyDeviceToHost);

    // for (int k = 0; k < Ksize; k++)
    // {
    //   for (int i = 0; i < Cin; i++)
    //   {
    //     for (int j = 0; j < Cout; j++)
    //       printf("%f ", hC[k*Ksize + i*Cout + j]);
    //     printf("\n");
    //   }
    //   printf("\n");
    // }


  }






void Conv3d_forward_cuda(
    point_data* d_Proot,
    uchar*      dO,
    uchar*      dE,
    uint*       dP,
    float*      InputPtr, int Cin,
    float*      OuputPtr, int Cout,
    float*      Weights,
    float*      Transposed_Weights,
    point_data* Kvec, uint Ksize,
    int         Jump,
    int         InLevel,
    int         OctreeLevel,
    int         BatchSize,
    uint*       Pyramid,
    uint*       d_Info,
    uint*       d_PSum,
    void*       d_temp_storageA,
    long        temp_storage_bytesA,
    int*        d_Inmap,
    int*        d_Outmap,
    int*        d_InmapX,
    int*        d_OutmapX,
    float*      d_EmptyIndicator,
    float*      d_UnseenIndicator,
    float*      d_Qcinout, 
    uint        insize) {

  pInOutMaps<int32_t>     d_inmap;
  pInOutMaps<int32_t>     d_outmap;

  float* Inputs = InputPtr;
  float* Outputs = OuputPtr;

  int OutLevel = InLevel - Jump;
  uint scale_factor = 0x1 << (InLevel - OutLevel);

  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  cublasSetStream(handle, stream);

  for (uint batch = 0; batch < BatchSize; batch++) {
    uint OutSize = GetPyramid(Pyramid, batch, 0, OutLevel, OctreeLevel);
    uint InSize = GetPyramid(Pyramid, batch, 0, InLevel, OctreeLevel);
    uint offset = GetPyramid(Pyramid, batch, 1, InLevel, OctreeLevel);

    GenerateKernelMap<<<(OutSize + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK,
                        THREADS_PER_BLOCK>>>(
        OutSize,
        d_Proot + GetPyramid(Pyramid, batch, 1, OutLevel, OctreeLevel),
        d_Inmap,
        d_Outmap,
        d_Info,
        Ksize, Kvec,
        scale_factor,
        dO, dE, dP, InLevel, offset,
        d_EmptyIndicator, d_UnseenIndicator);

    CUDA_CHECK(cudaGetLastError());

    ProcessKernelMaps(
        Ksize,
        OutSize,
        d_inmap,
        d_outmap,
        d_Info,
        d_PSum,
        d_temp_storageA,
        temp_storage_bytesA,
        d_Inmap,
        d_Outmap,
        d_InmapX,
        d_OutmapX);

    CUDA_CHECK(cudaGetLastError());

   minkowski::ConvolutionForwardKernelGPU<float, int32_t>(
        Inputs, Cin,// input
        Outputs, Cout,
        Weights, d_inmap, d_outmap, OutSize,
        handle, stream);

    CUDA_CHECK(cudaGetLastError());

    //empty space
    UnoccupiedContribution(
        handle,
        OutSize, Ksize, Cin, Cout,
        Transposed_Weights,
        d_EmptyIndicator,
        d_Qcinout,
        Outputs,
        InputPtr+insize);

    CUDA_CHECK(cudaGetLastError());

    // unseen space
    UnoccupiedContribution(
        handle,
        OutSize, Ksize, Cin, Cout,
        Transposed_Weights,
        d_UnseenIndicator,
        d_Qcinout,
        Outputs,
        InputPtr+insize+1);

    CUDA_CHECK(cudaGetLastError());

    Inputs += Cin * InSize;
    Outputs += Cout * OutSize;

    d_Proot += GetPyramid(Pyramid, batch, 1, OctreeLevel + 1, OctreeLevel);
    dO += GetPyramid(Pyramid, batch, 1, OctreeLevel, OctreeLevel);
    dE += GetPyramid(Pyramid, batch, 1, OctreeLevel, OctreeLevel);
    dP += GetPyramid(Pyramid, batch, 1, OctreeLevel, OctreeLevel) + 1;
  }
}

void Conv3d_backward_cuda(
    point_data*  d_Proot,
    uchar*     dO,
    uchar*     dE,
    uint*     dP,
    float*     InputPtr, int Cin,
    float*     Grad_Inputs,
    float*     Grad_Outputs, int Cout,
    float*     Weights, 
    float*      Transposed_Weights,
    float* Grad_Weights,
    point_data* Kvec, uint Ksize,
    int     Jump,
    int     OutLevel,
    int     OctreeLevel,
    int     BatchSize,
    uint*    Pyramid,
    uint*    d_Info,
    uint*    d_PSum,
    void*    d_temp_storageA,
    long    temp_storage_bytesA,
    int*    d_Inmap,
    int*    d_Outmap,
    int*    d_InmapX,
    int*    d_OutmapX,
    float*      d_EmptyIndicator,
    float*      d_UnseenIndicator,
    float*      d_Qcincout,
    float*      d_Kcout,
    uint        insize,
    uint        outsize) {
  pInOutMaps<int32_t> d_inmap;
  pInOutMaps<int32_t> d_outmap;

// printf("bkp:  %d   %d %d %d %d   %d %d\n", Ksize, OutLevel, Jump, Cin, Cout, insize, outsize);

  float* Inputs = InputPtr;

  int InLevel = OutLevel + Jump;
  uint scale_factor = 0x1 << (InLevel - OutLevel);

  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  cublasSetStream(handle, stream);

  for (uint batch = 0; batch < BatchSize; batch++) {
    uint InSize = GetPyramid(Pyramid, batch, 0, InLevel, OctreeLevel);
    uint OutSize = GetPyramid(Pyramid, batch, 0, OutLevel, OctreeLevel);
    uint offset = GetPyramid(Pyramid, batch, 1, OutLevel, OctreeLevel);

    GenerateKernelMapTrans<<<(InSize + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK,
                             THREADS_PER_BLOCK>>>(
        InSize,
        d_Proot + GetPyramid(Pyramid, batch, 1, InLevel, OctreeLevel),
        d_Inmap,
        d_Outmap,
        d_Info,
        Ksize, Kvec,
        scale_factor,
        dO, dE, dP, OutLevel, offset,
        d_EmptyIndicator, d_UnseenIndicator);

    CUDA_CHECK(cudaGetLastError());

    ProcessKernelMaps(
        Ksize,
        InSize,
        d_inmap,
        d_outmap,
        d_Info,
        d_PSum,
        d_temp_storageA,
        temp_storage_bytesA,
        d_Inmap,
        d_Outmap,
        d_InmapX,
        d_OutmapX);

    minkowski::ConvolutionBackwardKernelGPU<float, int32_t>(
        Inputs, Grad_Inputs, Cin,
        Grad_Outputs, Cout,
        Weights, Grad_Weights,
        d_outmap, d_inmap, OutSize, // note the swapping of i/o maps
        handle, stream);
    CUDA_CHECK(cudaGetLastError());





    //////////////////////////////////////////////////////////////
    //empty space
    UnoccupiedContribution(
      handle,
      InSize, Ksize, Cout, Cin,
      Weights,
      d_EmptyIndicator,
      d_Qcincout,
      Grad_Inputs,
      Grad_Outputs+outsize);

    CUDA_CHECK(cudaGetLastError());

    //unseen space
    UnoccupiedContribution(
      handle,
      InSize, Ksize, Cout, Cin,
      Weights,
      d_UnseenIndicator,
      d_Qcincout,
      Grad_Inputs,
      Grad_Outputs+outsize+1);
    CUDA_CHECK(cudaGetLastError());





    ///////WEIGHT UPDATE//////////////////////////////

    // need non-transposed neighbors
    // offset = GetPyramid(Pyramid, batch, 1, InLevel, OctreeLevel);
    // GenerateKernelMap2<<<(OutSize + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK,
    //                     THREADS_PER_BLOCK>>>(
    //     OutSize,
    //     d_Proot + GetPyramid(Pyramid, batch, 1, OutLevel, OctreeLevel),
    //     Ksize, Kvec,
    //     scale_factor,
    //     dO, dE, dP, InLevel, offset,
    //     d_EmptyIndicator, d_UnseenIndicator);

    // CUDA_CHECK(cudaGetLastError());


    // //empty weight update
    // UnoccupiedWeightUpdate(
    //   handle,
    //   OutSize, Ksize, Cout, Cin,
    //   Grad_Weights,
    //   d_EmptyIndicator,
    //   d_Kcout,
    //   Grad_Outputs,
    //   InputPtr+insize);
    // CUDA_CHECK(cudaGetLastError());

    // //unseen weight update
    // UnoccupiedWeightUpdate(
    //   handle,
    //   OutSize, Ksize, Cout, Cin,
    //   Grad_Weights,
    //   d_UnseenIndicator,
    //   d_Kcout,
    //   Grad_Outputs,
    //   InputPtr+insize+1);
    // CUDA_CHECK(cudaGetLastError());


    Inputs += Cin * InSize;
    Grad_Inputs += Cin * InSize;
    Grad_Outputs += Cout * OutSize;

    d_Proot += GetPyramid(Pyramid, batch, 1, OctreeLevel + 1, OctreeLevel);
    dO += GetPyramid(Pyramid, batch, 1, OctreeLevel, OctreeLevel);
    dE += GetPyramid(Pyramid, batch, 1, OctreeLevel, OctreeLevel);
    dP += GetPyramid(Pyramid, batch, 1, OctreeLevel, OctreeLevel) + 1;
  }


  CUDA_CHECK(cudaGetLastError());

}



void ConvTranspose3d_forward_cuda(
    point_data* d_Proot,
    uchar*      dO,
    uchar*      dE,
    uint*       dP,
    float*      InputPtr, int Cin,
    float*      OuputPtr, int Cout,
    float*      Weights,
    float*      Transposed_Weights,
    point_data* Kvec, uint Ksize,
    int         Jump,
    int         InLevel,
    int         OctreeLevel,
    int         BatchSize,
    uint*       Pyramid,
    uint*       d_Info,
    uint*       d_PSum,
    void*       d_temp_storageA,
    long        temp_storage_bytesA,
    int*        d_Inmap,
    int*        d_Outmap,
    int*        d_InmapX,
    int*        d_OutmapX,
    float*      d_EmptyIndicator,
    float*      d_UnseenIndicator,
    float*      d_Qcinout, 
    uint        insize) {

  pInOutMaps<int32_t>     d_inmap;
  pInOutMaps<int32_t>     d_outmap;

  float* Inputs = InputPtr;
  float* Outputs = OuputPtr;

  int OutLevel = InLevel + Jump;
  uint scale_factor = 0x1 << (OutLevel - InLevel);

  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  cublasSetStream(handle, stream);

  for (uint batch = 0; batch < BatchSize; batch++) {
    uint InSize = GetPyramid(Pyramid, batch, 0, InLevel, OctreeLevel);
    uint OutSize = GetPyramid(Pyramid, batch, 0, OutLevel, OctreeLevel);
    uint offset = GetPyramid(Pyramid, batch, 1, InLevel, OctreeLevel);

    GenerateKernelMapTrans<<<(OutSize + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK,
                             THREADS_PER_BLOCK>>>(
        OutSize,
        d_Proot + GetPyramid(Pyramid, batch, 1, OutLevel, OctreeLevel),
        d_Inmap,
        d_Outmap,
        d_Info,
        Ksize, Kvec,
        scale_factor,
        dO, dE, dP, InLevel, offset,
        d_EmptyIndicator, d_UnseenIndicator);

    CUDA_CHECK(cudaGetLastError());

    ProcessKernelMaps(
        Ksize,
        OutSize,
        d_inmap,
        d_outmap,
        d_Info,
        d_PSum,
        d_temp_storageA,
        temp_storage_bytesA,
        d_Inmap,
        d_Outmap,
        d_InmapX,
        d_OutmapX);

    minkowski::ConvolutionForwardKernelGPU<float, int32_t>(
        Inputs, Cin,// input
        Outputs, Cout,
        Weights, d_inmap, d_outmap, OutSize,
        handle, stream);

    CUDA_CHECK(cudaGetLastError());

    //empty space
    UnoccupiedContribution(
        handle,
        OutSize, Ksize, Cin, Cout,
        Transposed_Weights,
        d_EmptyIndicator,
        d_Qcinout,
        Outputs,
        InputPtr+insize);

    CUDA_CHECK(cudaGetLastError());

    //unseen space
    UnoccupiedContribution(
        handle,
        OutSize, Ksize, Cin, Cout,
        Transposed_Weights,
        d_UnseenIndicator,
        d_Qcinout,
        Outputs,
        InputPtr+insize+1);

    CUDA_CHECK(cudaGetLastError());

    Inputs += Cin * InSize;
    Outputs += Cout * OutSize;
 
    d_Proot += GetPyramid(Pyramid, batch, 1, OctreeLevel + 1, OctreeLevel);
    dO += GetPyramid(Pyramid, batch, 1, OctreeLevel, OctreeLevel);
    dE += GetPyramid(Pyramid, batch, 1, OctreeLevel, OctreeLevel);
    dP += GetPyramid(Pyramid, batch, 1, OctreeLevel, OctreeLevel) + 1;
  }
}


void ConvTranspose3d_backward_cuda(
    point_data*  d_Proot,
    uchar*     dO,
    uchar*     dE,
    uint*     dP,
    float*     InputPtr, int Cin,
    float*     Grad_Inputs,
    float*     Grad_Outputs, int Cout,
    float*     Weights, 
    float*      Transposed_Weights,
    float* Grad_Weights,
    point_data* Kvec, uint Ksize,
    int     Jump,
    int     OutLevel,
    int     OctreeLevel,
    int     BatchSize,
    uint*    Pyramid,
    uint*    d_Info,
    uint*    d_PSum,
    void*    d_temp_storageA,
    long    temp_storage_bytesA,
    int*    d_Inmap,
    int*    d_Outmap,
    int*    d_InmapX,
    int*    d_OutmapX,
    float*      d_EmptyIndicator,
    float*      d_UnseenIndicator,
    float*      d_Qcincout,
    float*      d_Kcout,
    uint        insize,
    uint        outsize) {
  pInOutMaps<int32_t>     d_inmap;
  pInOutMaps<int32_t>     d_outmap;

  float* Inputs = InputPtr;

  int InLevel = OutLevel - Jump;
  uint scale_factor = 0x1 << (OutLevel - InLevel);

  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  cublasSetStream(handle, stream);

  for (uint batch = 0; batch < BatchSize; batch++) {
    uint InSize = GetPyramid(Pyramid, batch, 0, InLevel, OctreeLevel);
    uint OutSize = GetPyramid(Pyramid, batch, 0, OutLevel, OctreeLevel);
    uint offset = GetPyramid(Pyramid, batch, 1, OutLevel, OctreeLevel);

    GenerateKernelMap<<<(InSize + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK,
                        THREADS_PER_BLOCK>>>(
        InSize,
        d_Proot + GetPyramid(Pyramid, batch, 1, InLevel, OctreeLevel),
        d_Inmap,
        d_Outmap,
        d_Info,
        Ksize, Kvec,
        scale_factor,
        dO, dE, dP, OutLevel, offset,
        d_EmptyIndicator, d_UnseenIndicator);

    CUDA_CHECK(cudaGetLastError());

    ProcessKernelMaps(
        Ksize,
        InSize,
        d_inmap,
        d_outmap,
        d_Info,
        d_PSum,
        d_temp_storageA,
        temp_storage_bytesA,
        d_Inmap,
        d_Outmap,
        d_InmapX,
        d_OutmapX);

    minkowski::ConvolutionBackwardKernelGPU<float, int32_t>(
        Inputs, Grad_Inputs, Cin,
        Grad_Outputs, Cout,
        Weights, Grad_Weights,
        d_outmap, d_inmap, OutSize,
        handle, stream);

    CUDA_CHECK(cudaGetLastError());



    //empty space
    // UnoccupiedContribution(
    //     handle,
    //     InSize, Ksize, Cout, Cin,
    //     Weights,
    //     d_EmptyIndicator,
    //     d_Qcincout,
    //     Grad_Inputs,
    //     Grad_Outputs+outsize);
    // CUDA_CHECK(cudaGetLastError());

    // //unseen space
    // UnoccupiedContribution(
    //     handle,
    //     InSize, Ksize, Cout, Cin,
    //     Weights,
    //     d_UnseenIndicator,
    //     d_Qcincout,
    //     Grad_Inputs,
    //     Grad_Outputs+outsize+1);
    // CUDA_CHECK(cudaGetLastError());



    /////////////////////////////////////////////////////////////////

    // need transposed neighbors
    // offset = GetPyramid(Pyramid, batch, 1, InLevel, OctreeLevel);
    // GenerateKernelMapTrans2<<<(OutSize + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK,
    //                     THREADS_PER_BLOCK>>>(
    //     OutSize,
    //     d_Proot + GetPyramid(Pyramid, batch, 1, OutLevel, OctreeLevel),
    //     Ksize, Kvec,
    //     scale_factor,
    //     dO, dE, dP, InLevel, offset,
    //     d_EmptyIndicator, d_UnseenIndicator);

    // CUDA_CHECK(cudaGetLastError());


    // //empty weight update
    // UnoccupiedWeightUpdate(
    //   handle,
    //   OutSize, Ksize, Cout, Cin,
    //   Grad_Weights,
    //   d_EmptyIndicator,
    //   d_Kcout,
    //   Grad_Outputs,
    //   InputPtr+insize);
    // CUDA_CHECK(cudaGetLastError());

    // //unseen weight update
    // UnoccupiedWeightUpdate(
    //   handle,
    //   OutSize, Ksize, Cout, Cin,
    //   Grad_Weights,
    //   d_UnseenIndicator,
    //   d_Kcout,
    //   Grad_Outputs,
    //   InputPtr+insize+1);
    // CUDA_CHECK(cudaGetLastError());





    Inputs += Cin * InSize;
    Grad_Inputs += Cin * InSize;
    Grad_Outputs += Cout * OutSize;

    d_Proot += GetPyramid(Pyramid, batch, 1, OctreeLevel + 1, OctreeLevel);
    dO += GetPyramid(Pyramid, batch, 1, OctreeLevel, OctreeLevel);
    dE += GetPyramid(Pyramid, batch, 1, OctreeLevel, OctreeLevel);
    dP += GetPyramid(Pyramid, batch, 1, OctreeLevel, OctreeLevel) + 1;
  }
}

}  // namespace kaolin
