/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#if defined _WIN32 || defined __APPLE__
#else
#define _LINUX
#endif

#include "paddle/fluid/framework/data_feed.h"
#include "paddle/fluid/framework/fleet/box_wrapper.h"

namespace paddle {
namespace framework {
using platform::PADDLE_CUDA_NUM_THREADS;

#define CUDA_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

__global__ void CopyForTensorKernel(FeatureItem* src, void** dest,
                                    size_t* offset, char* type,
                                    size_t total_size, size_t row_size,
                                    size_t col_size) {
  CUDA_KERNEL_LOOP(i, row_size * col_size) {
    int row_id = i / col_size;
    int col_id = i % col_size;
    size_t left, right;
    if (row_id == 0) {
      left = offset[row_id * (col_size + 1) + col_id];
      right = offset[row_id * (col_size + 1) + col_id + 1];
    } else {
      left = offset[row_id * (col_size + 1) + col_id] -
             offset[(row_id - 1) * (col_size + 1) + col_id];
      right = offset[row_id * (col_size + 1) + col_id + 1] -
              offset[(row_id - 1) * (col_size + 1) + col_id + 1];
    }
    uint64_t* up = NULL;
    float* fp = NULL;
    if (type[row_id] == 'f') {
      fp = reinterpret_cast<float*>(dest[row_id]);
    } else {
      up = reinterpret_cast<uint64_t*>(
          *(reinterpret_cast<uint64_t**>(dest) + row_id));
    }
    size_t begin = offset[row_id * (col_size + 1) + col_id + 1] +
                   offset[(row_size - 1) * (col_size + 1) + col_id] -
                   offset[row_id * (col_size + 1) + col_id] - (right - left);
    PADDLE_ENFORCE(begin >= 0, "begin must be ge 0.");
    PADDLE_ENFORCE(
        begin < total_size,
        "begin must be lt total_size, but your begin[%lu], total_size[%lu]",
        begin, total_size);
    for (size_t k = left; k < right; ++k) {
      PADDLE_ENFORCE((begin + k - left) >= 0 && (begin + k - left) < total_size,
                     "begin+k-left must be in [0, total_size)");
      if (type[row_id] == 'f') {
        *(fp + k) = src[begin + k - left].sign().float_feasign_;
      } else {
        *(up + k) = src[begin + k - left].sign().uint64_feasign_;
      }
    }
  }
}

void MultiSlotInMemoryDataFeed::CopyForTensor(
    const paddle::platform::Place& place, FeatureItem* src, void** dest,
    size_t* offset, char* type, size_t total_size, size_t row_size,
    size_t col_size) {
  auto stream = dynamic_cast<platform::CUDADeviceContext*>(
                    platform::DeviceContextPool::Instance().Get(
                        boost::get<platform::CUDAPlace>(place)))
                    ->stream();
  CopyForTensorKernel<<<((row_size * (col_size - 1)) + 511) / 512, 512, 0,
                        stream>>>(src, dest, offset, type, total_size, row_size,
                                  col_size - 1);
  cudaStreamSynchronize(stream);
}

__global__ void ExtractFeasign(uint64_t* dest, FeatureItem* src, size_t total_size) {
  CUDA_KERNEL_LOOP(i, total_size) {
    dest[i] = src[i].sign().uint64_feasign_;
  }
}

__global__ void SequencePoolKernel(float **dest, FeatureItem *src, float *emb, size_t *offset, int bs, int hidden_size) {
  CUDA_KERNEL_LOOP(i, total_size) {
    int low = 0;
    int high = bs - 1;
    while (low < high) {
      int mid = (low + high) / 2;
      if (i < len[mid])
        high = mid;
      else
        low = mid + 1;
    }
    int ins_id = low;
    uint16_t slot_id = src[i].slot();
    for (int j = 0; j < hidden_size; ++j) {
      paddle::platform::CudaAtomicAdd(dest[slot_id] + ins_id * hidden_size + j, emb[i * hidden_size + j]);
    }
  }
}

void MultiSlotInMemoryDataFeed::FuseCopyForTensor(
    const paddle::platform::Place& place, FeatureItem* src, float** dest,
    size_t* offset, size_t total_size, size_t bs) {
  auto stream = dynamic_cast<platform::CUDADeviceContext*>(platform::DeviceContextPool::Instance().Get(
                        boost::get<platform::CUDAPlace>(place)))
                    ->stream();
                    
  auto key_gpu_buf = memory::AllocShared(this->GetPlace(), total_size * sizeof(uint64_t));
  auto emb_gpu_buf = memory::AllocShared(this->GetPlace(), 11 * total_size * sizeof(float));

  uint64_t* key_gpu_data = reinterpret_cast<uint64_t>(key_gpu_buf->ptr());
  float* emb_gpu_data = reinterpret_cast<float>(emb_gpu_buf->ptr());
  
  ExtractFeasign<<<(total_size + PADDLE_CUDA_NUM_THREADS - 1) / PADDLE_CUDA_NUM_THREADS, PADDLE_CUDA_NUM_THREADS, 0 ,stream>>>(key_gpu_data, src, total_size);
  cudaStreamSynchronize(stream);

  // call PushSparse

  auto box_ptr = paddle::framework::BoxWrapper::GetInstance();
  std::vector<const uint64_t*> all_keys(1, key_gpu_data);
  std::vector<float *> all_values(1, emb_gpu_data);
  std::vector<int64_t> slot_lengths(1, total_size);
  box_ptr->PullSparse(place, all_keys, all_values, slot_lengths, 11);

  // SequencePool
  SequencePoolKernel<<<(total_size + PADDLE_CUDA_NUM_THREADS - 1) / PADDLE_CUDA_NUM_THREADS, PADDLE_CUDA_NUM_THREADS, 0 ,stream>>>(dest, emb_gpu_data, offset, bs, 11);
  cudaStreamSynchronize(stream);
}

}  // namespace framework
}  // namespace paddle
