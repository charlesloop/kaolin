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


#include "../../check.h"

#include <iostream>

#ifdef WITH_CUDA
#include "../../spc_math.h"
#endif

#include <ATen/ATen.h>

namespace kaolin {

#define CHECK_TRIPLE(x) TORCH_CHECK(x.dim() == 2 && x.size(1) == 3, #x " is not Nx3")

#define CHECK_OCTREES(x) CHECK_BYTE(x); CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_POINTS(x) CHECK_SHORT(x); CHECK_TRIPLE(x); CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_INPUT(x) CHECK_FLOAT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

using namespace at::indexing;

#ifdef WITH_CUDA
ulong GetStorageBytes(void* d_temp_storage, uint* d_Info, uint* d_PrefixSum, uint max_total_points);

void Conv3d_forward_cuda(
    point_data*  d_Proot,
    uchar*     dO,
    uchar*     dE,
    uint*     dP,
    float*     Inputs, int Cin,
    float*     Outputs, int Cout,
    float*      Weights,
    float*      Transposed_Weights,
    point_data* Kvec, uint Ksize,
    int     Jump,
    int     InLevel,
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
    float*      d_qcincout,
    uint        outsize);

void Conv3d_backward_cuda(
    point_data*  d_Proot,
    uchar*     dO,
    uchar*     dE,
    uint*     dP,
    float*     Inputs, int Cin,
    float*     Grad_Inputs,
    float*     Grad_Outputs, int Cout,
    float*      Weights,
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
    float*      d_qcincout,
    float*      d_qkcout,
    uint        insize,
    uint        outsize);

void ConvTranspose3d_forward_cuda(
    point_data*  d_Proot,
    uchar*     dO,
    uchar*     dE,
    uint*     dP,
    float*     Inputs, int Cin,
    float*     Outputs, int Cout,
    float*      Weights,
    float*      Transposed_Weights,
    point_data* Kvec, uint Ksize,
    int     Jump,
    int     InLevel,
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
    float*      d_qcincout,
    uint        outsize);

void ConvTranspose3d_backward_cuda(
    point_data*  d_Proot,
    uchar*     dO,
    uchar*     dE,
    uint*     dP,
    float*     Inputs, int Cin,
    float*     Grad_Inputs,
    float*     Grad_Outputs, int Cout,
    float*      Weights,
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
    float*      d_qcincout,
    float*      d_qkcout,
    uint        insize,
    uint        outsize);

#endif

std::tuple<at::Tensor, int> Conv3d_forward_empty(
    at::Tensor octree,
    at::Tensor points,
    uint level,
    at::Tensor pyramid,
    at::Tensor exsum,
    at::Tensor inputs,
    at::Tensor params,
    at::Tensor kernel_vectors,
    uint jump,
    at::Tensor empty) {
#ifdef WITH_CUDA
  CHECK_OCTREES(octree);
  CHECK_INPUT(inputs);
  CHECK_INPUT(params);
  CHECK_CONTIGUOUS(points);
  CHECK_CONTIGUOUS(pyramid);
  CHECK_CONTIGUOUS(kernel_vectors);
  CHECK_CUDA(points);
  CHECK_CPU(pyramid);
  CHECK_CUDA(kernel_vectors);
  CHECK_CUDA(empty);

  TORCH_CHECK(octree.size(0) == empty.size(0), "octree/empty size mismatch");

  uint kernel_vectors_size = params.size(0);
  TORCH_CHECK(kernel_vectors_size == kernel_vectors.size(0));
  point_data* Kvec = (point_data*)kernel_vectors.data_ptr<short>();

  uint Cin = params.size(1);
  TORCH_CHECK(Cin == inputs.size(1), "Wrong input channel size.");

  uint Cout = params.size(2);

  int BatchSize = pyramid.size(0);
  uint* Pyramid = reinterpret_cast<uint*>(pyramid.data_ptr<int>());

  int InLevel = level;
  int OutLevel = InLevel - jump;
  int OctreeLevel = pyramid.size(2)-2;
  TORCH_CHECK(OutLevel >= 0, "Illegal jump input, resulting in level underflow");

  uint insize = pyramid.index({ Slice(None), 0, InLevel }).sum().item<int>();
  uint outsize = pyramid.index({ Slice(None), 0, OutLevel }).sum().item<int>();
  int outmax = pyramid.index({ Slice(None), 0, OutLevel }).max().item<int>();
  TORCH_CHECK(inputs.size(0) == insize+2, "Bad input size.")

  at::Tensor outputs = at::zeros({ outsize + 2, Cout }, octree.options().dtype(at::kFloat)); // add 2 for empty/unseen

  auto transpose_weights = params.transpose(1,2);
  float* Weights = params.data_ptr<float>();
  float* Transpose_Weights = transpose_weights.data_ptr<float>();

  float*   Outputs = outputs.data_ptr<float>();
  float*   Inputs = inputs.data_ptr<float>();

  //intermediate storage
  int scan_size = kernel_vectors_size * outmax;

  // allocate local GPU storage
  at::Tensor Info = at::zeros({ scan_size }, octree.options().dtype(at::kInt));
  at::Tensor PrefixSum = at::zeros({ scan_size }, octree.options().dtype(at::kInt));
  at::Tensor Imap = at::zeros({ scan_size }, octree.options().dtype(at::kInt));
  at::Tensor Omap = at::zeros({ outmax }, octree.options().dtype(at::kInt));
  at::Tensor ImapX = at::zeros({ scan_size }, octree.options().dtype(at::kInt));
  at::Tensor OmapX = at::zeros({ scan_size }, octree.options().dtype(at::kInt));

  at::Tensor Etmp = at::zeros({ scan_size }, octree.options().dtype(at::kFloat));
  at::Tensor Utmp = at::zeros({ scan_size }, octree.options().dtype(at::kFloat));
  at::Tensor QCinCout = at::zeros({ outmax *  Cout * Cin }, octree.options().dtype(at::kFloat));

  // get tensor data pointers
  uint*  d_Info = reinterpret_cast<uint*>(Info.data_ptr<int>());
  uint*  d_PrefixSum = reinterpret_cast<uint*>(PrefixSum.data_ptr<int>());

  void* d_temp_storage = NULL;
  ulong temp_storage_bytes = GetStorageBytes(d_temp_storage, d_Info, d_PrefixSum, scan_size);
  at::Tensor temp_storage = at::zeros({ (long)temp_storage_bytes }, octree.options());
  d_temp_storage = (void*)temp_storage.data_ptr<uchar>();

  int* inmap = Imap.data_ptr<int>();
  int* outmap = Omap.data_ptr<int>();
  int* inmapX = ImapX.data_ptr<int>();
  int* outmapX = OmapX.data_ptr<int>();

  point_data* d_Proot = reinterpret_cast<point_data*>(points.data_ptr<short>());
  uchar* dO = octree.data_ptr<uchar>();
  uint* dEx = reinterpret_cast<uint*>(exsum.data_ptr<int>());
  uchar* dE = empty.data_ptr<uchar>();

  float* d_empty = Etmp.data_ptr<float>();
  float* d_useen = Utmp.data_ptr<float>();
  float* d_qcincout = QCinCout.data_ptr<float>();

  Conv3d_forward_cuda(
    d_Proot, dO, dE, dEx,
    Inputs, Cin, Outputs, Cout, 
    Weights, Transpose_Weights,
    Kvec, kernel_vectors_size, jump,
    InLevel, OctreeLevel, BatchSize, Pyramid,
    d_Info,
    d_PrefixSum,
    d_temp_storage,
    temp_storage_bytes,
    inmap,
    outmap,
    inmapX,
    outmapX,
    d_empty,
    d_useen,
    d_qcincout,
    insize);

  //map E/U imputs to outputs
  auto W = params.sum(0);
  auto in = inputs.index({ Slice(insize, insize+2) , Slice(None) });
  auto out = at::mm(in, W);

  // for (int i = 0; i < in.size(0); i++)
  // {
  //   for (int j = 0; j < in.size(1); j++)
  //   {
  //     printf("%f ", in[i][j].item<float>());
  //   }
  // }
  // printf("\n");

  // for (int i = 0; i < out.size(0); i++)
  // {
  //   for (int j = 0; j < out.size(1); j++)
  //   {
  //     printf("%f ", out[i][j].item<float>());
  //   }
  // }
  // printf("\n");

  //copy out -> end of outputs
  outputs.index({Slice(-2, None)}) = out;
  // outputs.index({Slice(-2, None)}) = 1.0f;//at::full((2,Cout), 0.1);

  return std::tuple<at::Tensor, int>{outputs, OutLevel};
#else
  AT_ERROR("Conv3d_forward not built with CUDA");
#endif
}


std::vector<at::Tensor> Conv3d_backward_empty(
    at::Tensor octree,
    at::Tensor points,
    uint level,
    at::Tensor pyramid,
    at::Tensor exsum,
    at::Tensor inputs,
    at::Tensor grad_outputs,
    at::Tensor params,
    at::Tensor kernel_vectors,
    uint jump,
    at::Tensor empty) {
#ifdef WITH_CUDA
  CHECK_INPUT(grad_outputs);
  CHECK_INPUT(inputs);
  CHECK_INPUT(params);
  CHECK_CONTIGUOUS(points);
  CHECK_CONTIGUOUS(pyramid);
  CHECK_CONTIGUOUS(kernel_vectors);
  CHECK_CUDA(points);
  CHECK_CPU(pyramid);
  CHECK_CUDA(kernel_vectors);


  uint kernel_vectors_size = params.size(0);
  TORCH_CHECK(kernel_vectors_size == kernel_vectors.size(0));
  point_data* Kvec = (point_data*)kernel_vectors.data_ptr<short>();

  uint Cin = params.size(1);
  TORCH_CHECK(Cin == inputs.size(1));

  uint Cout = params.size(2);

  int BatchSize = pyramid.size(0);
  uint* Pyramid = reinterpret_cast<uint*>(pyramid.data_ptr<int>());

  int OutLevel = level;
  int InLevel = OutLevel + jump;
  int OctreeLevel = pyramid.size(2)-2;

  uint outsize = pyramid.index({ Slice(None), 0, OutLevel }).sum().item<int>();
  uint insize = pyramid.index({ Slice(None), 0, InLevel }).sum().item<int>();
  // int outmax = pyramid.index({ Slice(None), 0, OutLevel }).max().item<int>();
  int inmax = pyramid.index({ Slice(None), 0, InLevel }).max().item<int>();



  // printf("cnvbk: %d  %d  %d   %d %d    %d\n", level, Cin, Cout, insize, outsize, inmax);



  at::Tensor grad_inputs = at::zeros_like(inputs);
  at::Tensor grad_params = at::zeros_like(params);

  float* Weights = params.data_ptr<float>();
  auto transpose_weights = params.transpose(1,2);
  float* Transpose_Weights = transpose_weights.data_ptr<float>();

  float* grad_Weights = grad_params.data_ptr<float>();
  float* Grad_Outputs = grad_outputs.data_ptr<float>();
  float* Grad_Inputs = grad_inputs.data_ptr<float>();
  float* Inputs = inputs.data_ptr<float>();


  //intermediate storage
  int scan_size = kernel_vectors_size * inmax;

  // allocate local GPU storage
  at::Tensor Info = at::zeros({ scan_size }, octree.options().dtype(at::kInt));
  at::Tensor PrefixSum = at::zeros({ scan_size }, octree.options().dtype(at::kInt));
  at::Tensor Imap = at::zeros({ scan_size }, octree.options().dtype(at::kInt));
  at::Tensor Omap = at::zeros({ inmax }, octree.options().dtype(at::kInt));
  at::Tensor ImapX = at::zeros({ scan_size }, octree.options().dtype(at::kInt));
  at::Tensor OmapX = at::zeros({ scan_size }, octree.options().dtype(at::kInt));

  at::Tensor Etmp = at::zeros({ kernel_vectors_size * inmax }, octree.options().dtype(at::kFloat));
  at::Tensor Utmp = at::zeros({ kernel_vectors_size * inmax }, octree.options().dtype(at::kFloat));
  at::Tensor QCinCout = at::zeros({ inmax *  Cout * Cin }, octree.options().dtype(at::kFloat));
  at::Tensor KCout = at::zeros({ kernel_vectors_size * Cout }, octree.options().dtype(at::kFloat));

  // get tensor data pointers
  uint*  d_Info = reinterpret_cast<uint*>(Info.data_ptr<int>());
  uint*  d_PrefixSum = reinterpret_cast<uint*>(PrefixSum.data_ptr<int>());

  void* d_temp_storage = NULL;
  ulong temp_storage_bytes = GetStorageBytes(d_temp_storage, d_Info, d_PrefixSum, scan_size);
  at::Tensor temp_storage = at::zeros({ (long)temp_storage_bytes }, octree.options());
  d_temp_storage = (void*)temp_storage.data_ptr<uchar>();

  int* inmap = Imap.data_ptr<int>();
  int* outmap = Omap.data_ptr<int>();
  int* inmapX = ImapX.data_ptr<int>();
  int* outmapX = OmapX.data_ptr<int>();

  point_data*  d_Proot = (point_data*)points.data_ptr<short>();
  uchar*     dO = octree.data_ptr<uchar>();
  uint*     dEx = reinterpret_cast<uint*>(exsum.data_ptr<int>());
  uchar* dE = empty.data_ptr<uchar>();
  
  float* d_empty = Etmp.data_ptr<float>();
  float* d_useen = Utmp.data_ptr<float>();
  float* d_qcincout = QCinCout.data_ptr<float>();
  float* d_kcout = KCout.data_ptr<float>();

  Conv3d_backward_cuda(
    d_Proot, dO, dE, dEx,
    Inputs, Cin, Grad_Inputs, Grad_Outputs, Cout, 
    Weights, Transpose_Weights, grad_Weights, 
    Kvec, kernel_vectors_size, jump,
    OutLevel, OctreeLevel, BatchSize, Pyramid,
    d_Info,
    d_PrefixSum,
    d_temp_storage,
    temp_storage_bytes,
    inmap,
    outmap,
    inmapX,
    outmapX,
    d_empty,
    d_useen,
    d_qcincout,
    d_kcout,
    insize,
    outsize);

  //map E/U imputs to outputs
  auto W = transpose_weights.sum(0).div_(powf32(8.0, jump));
  auto in = grad_outputs.index({ Slice(outsize, outsize+2) , Slice(None) });
  auto out = at::mm(in , W);

  // std::cout << "Q" << InLevel << "  P" << OutLevel << "\n";
  // std::cout << in;
  // std::cout << out;

  //copy out -> end of outputs
  grad_inputs.index({Slice(-2, None)}) = out;
  // grad_inputs.index({Slice(-2, None)}) = 1.0f;//at::full((2,Cin), 0.1);


  return {grad_inputs, grad_params};
#else
  AT_ERROR("Conv3d_backward not built with CUDA");
#endif
}

std::tuple<at::Tensor, int> ConvTranspose3d_forward_empty(
    at::Tensor octree,
    at::Tensor points,
    uint level,
    at::Tensor pyramid,
    at::Tensor exsum,
    at::Tensor inputs,
    at::Tensor params,
    at::Tensor kernel_vectors,
    uint jump,
    at::Tensor empty) {
#if WITH_CUDA
  CHECK_OCTREES(octree);
  CHECK_INPUT(inputs);
  CHECK_INPUT(params);
  CHECK_CONTIGUOUS(points);
  CHECK_CONTIGUOUS(pyramid);
  CHECK_CONTIGUOUS(kernel_vectors);
  CHECK_CUDA(points);
  CHECK_CPU(pyramid);
  CHECK_CUDA(kernel_vectors);

  TORCH_CHECK(octree.size(0) == empty.size(0), "octree/empty size mismatch");

  uint kernel_vectors_size = params.size(0);
  TORCH_CHECK(kernel_vectors_size == kernel_vectors.size(0));
  point_data* Kvec = (point_data*)kernel_vectors.data_ptr<short>();

  uint Cin = params.size(1);
  TORCH_CHECK(Cin == inputs.size(1));

  uint Cout = params.size(2);

  int BatchSize = pyramid.size(0);
  uint* Pyramid = reinterpret_cast<uint*>(pyramid.data_ptr<int>());

  int InLevel = level;
  int OutLevel = InLevel + jump;
  int OctreeLevel = pyramid.size(2)-2;
  TORCH_CHECK(OutLevel <= OctreeLevel, "Illegal jump input, resulting in level overflow");

  uint insize = pyramid.index({ Slice(None), 0, InLevel }).sum().item<int>();
  uint outsize = pyramid.index({ Slice(None), 0, OutLevel }).sum().item<int>();
  int outmax = pyramid.index({ Slice(None), 0, OutLevel }).max().item<int>();
  TORCH_CHECK(inputs.size(0) == insize+2, "Bad input size.")

  at::Tensor outputs = at::zeros({ outsize + 2, Cout }, octree.options().dtype(at::kFloat)); // add 2 for empty/unseen

  auto transpose_weights = params.transpose(1,2);
  float* Weights = params.data_ptr<float>();
  float* Transpose_Weights = transpose_weights.data_ptr<float>();

  float*   Outputs = outputs.data_ptr<float>();
  float*   Inputs = inputs.data_ptr<float>();

  //intermediate storage
  int scan_size = kernel_vectors_size * outmax;

  // allocate local GPU storage
  at::Tensor Info = at::zeros({ scan_size }, octree.options().dtype(at::kInt));
  at::Tensor PrefixSum = at::zeros({ scan_size }, octree.options().dtype(at::kInt));
  at::Tensor Imap = at::zeros({ scan_size }, octree.options().dtype(at::kInt));
  at::Tensor Omap = at::zeros({ outmax }, octree.options().dtype(at::kInt));
  at::Tensor ImapX = at::zeros({ scan_size }, octree.options().dtype(at::kInt));
  at::Tensor OmapX = at::zeros({ scan_size }, octree.options().dtype(at::kInt));

  at::Tensor Etmp = at::zeros({ scan_size }, octree.options().dtype(at::kFloat));
  at::Tensor Utmp = at::zeros({ scan_size }, octree.options().dtype(at::kFloat));
  at::Tensor QCinCout = at::zeros({ outmax *  Cout * Cin }, octree.options().dtype(at::kFloat));

  // get tensor data pointers
  uint*  d_Info = reinterpret_cast<uint*>(Info.data_ptr<int>());
  uint*  d_PrefixSum = reinterpret_cast<uint*>(PrefixSum.data_ptr<int>());

  void* d_temp_storage = NULL;
  ulong temp_storage_bytes = GetStorageBytes(d_temp_storage, d_Info, d_PrefixSum, scan_size);
  at::Tensor temp_storage = at::zeros({ (long)temp_storage_bytes }, octree.options().dtype(at::kByte));
  d_temp_storage = (void*)temp_storage.data_ptr<uchar>();

  int* inmap = Imap.data_ptr<int>();
  int* outmap = Omap.data_ptr<int>();
  int* inmapX = ImapX.data_ptr<int>();
  int* outmapX = OmapX.data_ptr<int>();

  point_data*  d_Proot = (point_data*)points.data_ptr<short>();
  uchar*     dO = octree.data_ptr<uchar>();
  uint*     dEx = reinterpret_cast<uint*>(exsum.data_ptr<int>());
  uchar* dE = empty.data_ptr<uchar>();

  float* d_empty = Etmp.data_ptr<float>();
  float* d_useen = Utmp.data_ptr<float>();
  float* d_qcincout = QCinCout.data_ptr<float>();

  ConvTranspose3d_forward_cuda(
    d_Proot, dO, dE, dEx,
    Inputs, Cin, Outputs, Cout, 
    Weights, Transpose_Weights,
    Kvec, kernel_vectors_size, jump,
    InLevel, OctreeLevel, BatchSize, Pyramid,
    d_Info,
    d_PrefixSum,
    d_temp_storage,
    temp_storage_bytes,
    inmap,
    outmap,
    inmapX,
    outmapX,
    d_empty,
    d_useen,
    d_qcincout,
    insize);

  //map E/U imputs to outputs
  auto W = params.sum(0).div_(powf32(8.0, jump));
  auto in = inputs.index({ Slice(insize, insize+2) , Slice(None) });
  auto out = at::mm(in, W);

  // //copy out -> end of outputs
  outputs.index({Slice(-2, None)}) = out;
  // outputs.index({Slice(-2, None)}) = 1.0f;//at::full((2,Cout), 0.1);

  return std::tuple<at::Tensor, int>{outputs, OutLevel};
#else
  AT_ERROR("ConvTranspose3d_forward not built with CUDA");
#endif
}

std::vector<at::Tensor>  ConvTranspose3d_backward_empty(
    at::Tensor octree,
    at::Tensor points,
    uint level,
    at::Tensor pyramid,
    at::Tensor exsum,
    at::Tensor inputs,
    at::Tensor grad_outputs,
    at::Tensor params,
    at::Tensor kernel_vectors,
    uint jump,
    at::Tensor empty) {
#if WITH_CUDA
  CHECK_OCTREES(octree);
  CHECK_INPUT(grad_outputs);
  CHECK_INPUT(inputs);
  CHECK_INPUT(params);
  CHECK_CONTIGUOUS(points);
  CHECK_CONTIGUOUS(pyramid);
  CHECK_CONTIGUOUS(kernel_vectors);
  CHECK_CUDA(points);
  CHECK_CPU(pyramid);
  CHECK_CUDA(kernel_vectors);

  uint kernel_vectors_size = params.size(0);
  TORCH_CHECK(kernel_vectors_size == kernel_vectors.size(0));
  point_data* Kvec = (point_data*)kernel_vectors.data_ptr<short>();

  uint Cin = params.size(1);
  TORCH_CHECK(Cin == inputs.size(1));

  uint Cout = params.size(2);

  int BatchSize = pyramid.size(0);
  uint* Pyramid = reinterpret_cast<uint*>(pyramid.data_ptr<int>());

  int OutLevel = level;
  int InLevel = OutLevel - jump;
  int OctreeLevel = pyramid.size(2)-2;

  uint outsize = pyramid.index({ Slice(None), 0, OutLevel }).sum().item<int>();
  uint insize = pyramid.index({ Slice(None), 0, InLevel }).sum().item<int>();
  int outmax = pyramid.index({ Slice(None), 0, OutLevel }).max().item<int>();

  at::Tensor grad_inputs = at::zeros_like(inputs);
  at::Tensor grad_params = at::zeros_like(params);

  float* Weights = params.data_ptr<float>();
  auto transpose_weights = params.transpose(1,2);
  float* Transpose_Weights = transpose_weights.data_ptr<float>();

  float* grad_Weights = grad_params.data_ptr<float>();
  float* Grad_Outputs = grad_outputs.data_ptr<float>();
  float* Grad_Inputs = grad_inputs.data_ptr<float>();
  float* Inputs = inputs.data_ptr<float>();


  //intermediate storage
  int scan_size = kernel_vectors_size * outmax;

  // allocate local GPU storage
  at::Tensor Info = at::zeros({ scan_size }, octree.options().dtype(at::kInt));
  at::Tensor PrefixSum = at::zeros({ scan_size }, octree.options().dtype(at::kInt));
  at::Tensor Imap = at::zeros({ scan_size }, octree.options().dtype(at::kInt));
  at::Tensor Omap = at::zeros({ outmax }, octree.options().dtype(at::kInt));
  at::Tensor ImapX = at::zeros({ scan_size }, octree.options().dtype(at::kInt));
  at::Tensor OmapX = at::zeros({ scan_size }, octree.options().dtype(at::kInt));

  at::Tensor Etmp = at::zeros({ kernel_vectors_size * outmax }, octree.options().dtype(at::kFloat));
  at::Tensor Utmp = at::zeros({ kernel_vectors_size * outmax }, octree.options().dtype(at::kFloat));
  at::Tensor QCinCout = at::zeros({ outmax *  Cout * Cin }, octree.options().dtype(at::kFloat));
  at::Tensor KCout = at::zeros({ kernel_vectors_size * Cout }, octree.options().dtype(at::kFloat));

  // get tensor data pointers
  uint*  d_Info = reinterpret_cast<uint*>(Info.data_ptr<int>());
  uint*  d_PrefixSum = reinterpret_cast<uint*>(PrefixSum.data_ptr<int>());

  void* d_temp_storage = NULL;
  ulong temp_storage_bytes = GetStorageBytes(d_temp_storage, d_Info, d_PrefixSum, scan_size);
  at::Tensor temp_storage = at::zeros({ (long)temp_storage_bytes }, octree.options());
  d_temp_storage = (void*)temp_storage.data_ptr<uchar>();

  int* inmap = Imap.data_ptr<int>();
  int* outmap = Omap.data_ptr<int>();
  int* inmapX = ImapX.data_ptr<int>();
  int* outmapX = OmapX.data_ptr<int>();

  point_data*  d_Proot = (point_data*)points.data_ptr<short>();
  uchar*     dO = octree.data_ptr<uchar>();
  uint*     dEx = reinterpret_cast<uint*>(exsum.data_ptr<int>());
  uchar* dE = empty.data_ptr<uchar>();

  float* d_empty = Etmp.data_ptr<float>();
  float* d_useen = Utmp.data_ptr<float>();
  float* d_qcincout = QCinCout.data_ptr<float>();
  float* d_kcout = KCout.data_ptr<float>();

  ConvTranspose3d_backward_cuda(
    d_Proot, dO, dE, dEx,
    Inputs, Cin, Grad_Inputs, Grad_Outputs, Cout, 
    Weights, Transpose_Weights, grad_Weights, 
    Kvec, kernel_vectors_size, jump,
    OutLevel, OctreeLevel, BatchSize, Pyramid,
    d_Info,
    d_PrefixSum,
    d_temp_storage,
    temp_storage_bytes,
    inmap,
    outmap,
    inmapX,
    outmapX,
    d_empty,
    d_useen,
    d_qcincout,
    d_kcout,
    insize,
    outsize);

  //map E/U imputs to outputs
  auto W = transpose_weights.sum(0);//.div_(powf32(8.0, jump));
  auto in = grad_outputs.index({ Slice(outsize, outsize+2) , Slice(None) });
  auto out = at::mm(in, W);
 
  //copy out -> end of outputs
  grad_inputs.index({Slice(-2, None)}) = out;
  // grad_inputs.index({Slice(-2, None)}) = 1.0f;//at::full((2,Cin), 0.1);

  return {grad_inputs, grad_params};
#else
  AT_ERROR("ConvTranspose3d_backward not built with CUDA");
#endif
}

}  // namespace kaolin
