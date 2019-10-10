// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <ctime>
#include <memory>
#include <numeric>
#include "paddle/fluid/framework/fleet/box_wrapper.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/platform/gpu_info.h"

namespace paddle {
namespace framework {

std::shared_ptr<BoxWrapper> BoxWrapper::s_instance_ = nullptr;
#ifdef PADDLE_WITH_BOX_PS
cudaStream_t BoxWrapper::stream_list_[8];
std::shared_ptr<boxps::BoxPSBase> BoxWrapper::boxps_ptr_ = nullptr;
#endif

int BoxWrapper::GetDate() const {
  time_t now = time(0);
  tm t;
#ifdef _WIN32
  localtime_s(&t, &now);
#else
  localtime_r(&now, &t);
#endif
  char buf[10];
  snprintf(buf, sizeof(buf), "%04d%02d%02d", (1900 + t.tm_year), (1 + t.tm_mon),
           t.tm_mday);
  return atoi(buf);
}

#define CUDA_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

__global__ void PullCopy(float** dest, abacus::FeatureValueGpu* src,
                         int64_t* len, int hidden, int slot_num,
                         int total_len) {
  CUDA_KERNEL_LOOP(i, total_len) {
    int low = 0;
    int high = slot_num - 1;
    while (low < high) {
      int mid = (low + high) / 2;
      if (i < len[mid])
        high = mid;
      else
        low = mid + 1;
    }
    int x = low;
    int y = i - (x ? len[x - 1] : 0);
    *(dest[x] + y * hidden) = (src + i)->show;
    *(dest[x] + y * hidden + 1) = (src + i)->clk;
    *(dest[x] + y * hidden + 2) = (src + i)->embed_w;
    for (int j = 0; j < 8; j++) {
      *(dest[x] + y * hidden + 3 + j) = (src + i)->embedx[1 + j];
    }
  }
}

__global__ void PushCopy(abacus::FeaturePushValueGpu* dest, float** src,
                         int64_t* len, int hidden, int slot_num,
                         int total_len) {
  CUDA_KERNEL_LOOP(i, total_len) {
    int low = 0;
    int high = slot_num - 1;
    while (low < high) {
      int mid = (low + high) / 2;
      if (i < len[mid])
        high = mid;
      else
        low = mid + 1;
    }
    int x = low;
    int y = i - (x ? len[low - 1] : 0);
    (dest + i)->show = *(src[x] + y * hidden);
    (dest + i)->clk = *(src[x] + y * hidden + 1);
    (dest + i)->embed_g = *(src[x] + y * hidden + 2);
    for (int j = 0; j < 8; j++) {
      (dest + i)->embedx_g[j] = *(src[x] + y * hidden + 3 + j);
    }
  }
}

void BoxWrapper::ResetClickNum() {
  actual_click.store(0);
  std::lock_guard<std::mutex> lock(add_mutex);
  pred_click = 0.0;
}

void BoxWrapper::UpdateClickNum(int64_t actual_val, float pred_val) {
  actual_click.fetch_add(actual_val);
  std::lock_guard<std::mutex> lock(add_mutex);
  pred_click += pred_val;
}

void BoxWrapper::PrintClickNum() const {
  auto actual_val = actual_click.load();
  auto pred_val = pred_click;
  printf("actual click: %ld, pred click: %lf\n", actual_val, pred_val);
  printf("COPC: %lf\n", static_cast<float>(actual_val) / pred_val);
}

void BoxWrapper::FeedPass(const std::vector<uint64_t>& feasgin_to_box) const {
#ifdef PADDLE_WITH_BOX_PS
  int ret = boxps_ptr_->FeedPass(GetDate(), feasgin_to_box);
  PADDLE_ENFORCE_EQ(ret, 0, "FeedPass failed in BoxPS.");
#endif
}

void BoxWrapper::BeginPass() const {
#ifdef PADDLE_WITH_BOX_PS
  int ret = boxps_ptr_->BeginPass();
  PADDLE_ENFORCE_EQ(ret, 0, "BeginPass failed in BoxPS.");
#endif
}

void BoxWrapper::EndPass() const {
#ifdef PADDLE_WITH_BOX_PS
  int ret = boxps_ptr_->EndPass();
  PADDLE_ENFORCE_EQ(ret, 0, "EndPass failed in BoxPS.");
#endif
}

void BoxWrapper::PullSparse(const paddle::platform::Place& place,
                            const std::vector<const uint64_t*>& keys,
                            const std::vector<float*>& values,
                            const std::vector<int64_t>& slot_lengths,
                            const int hidden_size) {
  PADDLEBOX_LOG << "Begin call PullSparse in BoxWrapper";
  platform::Timer all_timer;
  platform::Timer pull_boxps_timer;
  all_timer.Start();
#ifdef PADDLE_WITH_BOX_PS
  if (platform::is_cpu_place(place) || platform::is_gpu_place(place)) {
    int64_t total_length =
        std::accumulate(slot_lengths.begin(), slot_lengths.end(), 0UL);
    LoDTensor total_keys_tensor;
    int64_t* total_keys =
        total_keys_tensor.mutable_data<int64_t>({total_length, 1}, place);

    int64_t offset = 0;
    VLOG(3) << "Begin copy keys, key_num[" << keys.size() << "]";
    for (size_t i = 0; i < keys.size(); ++i) {
      if (platform::is_cpu_place(place)) {
        memory::Copy(boost::get<platform::CPUPlace>(place), total_keys + offset,
                     boost::get<platform::CPUPlace>(place), keys[i],
                     slot_lengths[i] * sizeof(uint64_t));
      } else {
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
        memory::Copy(boost::get<platform::CUDAPlace>(place),
                     total_keys + offset,
                     boost::get<platform::CUDAPlace>(place), keys[i],
                     slot_lengths[i] * sizeof(uint64_t), nullptr);
#else
        PADDLE_THROW(
            "Please compile WITH_GPU option, and NCCL doesn't support "
            "windows.");
#endif
      }
      offset += slot_lengths[i];
    }
    VLOG(3) << "End copy keys";
    PADDLE_ENFORCE_EQ(offset, total_length,
                      "BoxWrapper::PullSparse: total feasign keys length "
                      "should be equal to the sum of length of all input "
                      "tensors.");

    // Space allocation for FeatureValue is left for boxps
    auto buf = memory::AllocShared(
        place, total_length * sizeof(abacus::FeatureValueGpu));
    abacus::FeatureValueGpu* total_values_gpu =
        reinterpret_cast<abacus::FeatureValueGpu*>(buf->ptr());
    VLOG(3) << "Begin PullSparseGPU";
    pull_boxps_timer.Start();
    if (platform::is_cpu_place(place)) {
      // TODO(hutuxian): should use boxps::FeatureValue in the future
      int ret = boxps_ptr_->PullSparseCPU(
          reinterpret_cast<uint64_t*>(total_keys), total_values_gpu,
          static_cast<int>(total_length));
      PADDLE_ENFORCE_EQ(ret, 0, "PullSparseCPU failed in BoxPS.");
    } else {
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
      int ret = boxps_ptr_->PullSparseGPU(
          reinterpret_cast<uint64_t*>(total_keys), total_values_gpu,
          static_cast<int>(total_length),
          boost::get<platform::CUDAPlace>(place).GetDeviceId());
      PADDLE_ENFORCE_EQ(ret, 0, "PullSparseGPU failed in BoxPS.");
      VLOG(3) << "End call boxps_ptr_->PullSparseGPU";
#endif
    }
    pull_boxps_timer.Pause();

    offset = 0;
    VLOG(3) << "Begin Copy result to tensor, total_length[" << total_length
            << "]";

    auto stream = dynamic_cast<platform::CUDADeviceContext*>(
                      platform::DeviceContextPool::Instance().Get(
                          boost::get<platform::CUDAPlace>(place)))
                      ->stream();
    auto slot_lengths_lod = slot_lengths;
    for (int i = 1; i < slot_lengths_lod.size(); i++) {
      slot_lengths_lod[i] += slot_lengths_lod[i - 1];
    }
    auto buf1 = memory::AllocShared(place, values.size() * sizeof(float*));
    auto buf2 =
        memory::AllocShared(place, slot_lengths.size() * sizeof(int64_t));
    float** gpu_values = reinterpret_cast<float**>(buf1->ptr());
    int64_t* gpu_len = reinterpret_cast<int64_t*>(buf2->ptr());
    cudaMemcpy(gpu_values, values.data(), values.size() * sizeof(float*),
               cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_len, slot_lengths_lod.data(),
               slot_lengths.size() * sizeof(int64_t), cudaMemcpyHostToDevice);
    PullCopy<<<(total_length + 512 - 1) / 512, 512, 0, stream>>>(
        gpu_values, total_values_gpu, gpu_len, hidden_size, slot_lengths.size(),
        total_length);
    cudaStreamSynchronize(stream);
    all_timer.Pause();

    PADDLEBOX_LOG << "End PullSparse in BoxWrapper: total cost: "
                  << all_timer.ElapsedSec() << " s, and pull boxps cost: "
                  << pull_boxps_timer.ElapsedSec() << " s";

    // only support gpu for paddlebox
    /*
    for (size_t i = 0; i < values.size(); ++i) {
      int64_t fea_num = slot_lengths[i];
      VLOG(3) << "Begin Copy slot[" << i << "] fea_num[" << fea_num << "]";
      for (auto j = 0; j < fea_num; ++j) {
        // Copy the emb from BoxPS to paddle tensor. Since 'show','click','emb'
        // are continuous in memory, so we copy here using the 'show' address
        if (platform::is_cpu_place(place)) {
          memory::Copy(
              boost::get<platform::CPUPlace>(place),
              values[i] + j * hidden_size,
              boost::get<platform::CPUPlace>(place),
              reinterpret_cast<float*>(&((total_values_gpu + offset)->show)),
              sizeof(float) * hidden_size);
        } else {
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
          memory::Copy(
              boost::get<platform::CUDAPlace>(place),
              values[i] + j * hidden_size,
              boost::get<platform::CUDAPlace>(place),
              reinterpret_cast<float*>(&((total_values_gpu + offset)->show)),
              sizeof(float) * hidden_size, nullptr);
#endif
        }
        ++offset;
      }
      VLOG(3) << "End Copy slot[" << i << "] fea_num[" << fea_num << "] offset["
              << offset << "]";
    }
                */
    VLOG(3) << "End Copy result to tensor";
  } else {
    PADDLE_THROW(
        "PaddleBox: PullSparse Only Support CPUPlace and CUDAPlace Now.");
  }
#endif
  VLOG(3) << "End call PullSparse";
}

void BoxWrapper::PushSparseGrad(const paddle::platform::Place& place,
                                const std::vector<const uint64_t*>& keys,
                                const std::vector<const float*>& grad_values,
                                const std::vector<int64_t>& slot_lengths,
                                const int hidden_size) {
  PADDLEBOX_LOG << "Begin call PushSparseGrad in BoxWrapper";
  platform::Timer all_timer;
  platform::Timer push_boxps_timer;
  all_timer.Start();
#ifdef PADDLE_WITH_BOX_PS
  if (platform::is_cpu_place(place) || platform::is_gpu_place(place)) {
    int64_t total_length =
        std::accumulate(slot_lengths.begin(), slot_lengths.end(), 0UL);
    LoDTensor total_keys_tensor;
    int64_t* total_keys =
        total_keys_tensor.mutable_data<int64_t>({total_length, 1}, place);
    int64_t offset = 0;
    for (size_t i = 0; i < keys.size(); ++i) {
      if (platform::is_cpu_place(place)) {
        memory::Copy(boost::get<platform::CPUPlace>(place), total_keys + offset,
                     boost::get<platform::CPUPlace>(place), keys[i],
                     slot_lengths[i] * sizeof(uint64_t));
      } else {
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
        memory::Copy(boost::get<platform::CUDAPlace>(place),
                     total_keys + offset,
                     boost::get<platform::CUDAPlace>(place), keys[i],
                     slot_lengths[i] * sizeof(uint64_t), nullptr);
#else
        PADDLE_THROW(
            "Please compile WITH_GPU option, and for now NCCL doesn't support "
            "windows.");
#endif
      }
      offset += slot_lengths[i];
    }
    PADDLE_ENFORCE_EQ(offset, total_length,
                      "BoxWrapper::PushSparseGrad: total feasign keys length "
                      "should be equal to the sum of length of all input "
                      "tensors.");
    auto buf = memory::AllocShared(
        place, total_length * sizeof(abacus::FeaturePushValueGpu));
    abacus::FeaturePushValueGpu* total_grad_values_gpu =
        reinterpret_cast<abacus::FeaturePushValueGpu*>(buf->ptr());

    offset = 0;

    auto stream = dynamic_cast<platform::CUDADeviceContext*>(
                      platform::DeviceContextPool::Instance().Get(
                          boost::get<platform::CUDAPlace>(place)))
                      ->stream();
    auto slot_lengths_lod = slot_lengths;
    for (int i = 1; i < slot_lengths_lod.size(); i++) {
      slot_lengths_lod[i] += slot_lengths_lod[i - 1];
    }
    auto buf1 = memory::AllocShared(place, grad_values.size() * sizeof(float*));
    auto buf2 =
        memory::AllocShared(place, slot_lengths.size() * sizeof(int64_t));
    float** gpu_values = reinterpret_cast<float**>(buf1->ptr());
    int64_t* gpu_len = reinterpret_cast<int64_t*>(buf2->ptr());

    cudaMemcpy(gpu_values, grad_values.data(),
               grad_values.size() * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_len, slot_lengths_lod.data(),
               slot_lengths.size() * sizeof(int64_t), cudaMemcpyHostToDevice);
    PushCopy<<<(total_length + 512 - 1) / 512, 512, 0, stream>>>(
        total_grad_values_gpu, gpu_values, gpu_len, hidden_size,
        slot_lengths.size(), total_length);
    cudaStreamSynchronize(stream);

    // only support gpu for paddlebox
    /*
    for (size_t i = 0; i < grad_values.size(); ++i) {
      int64_t fea_num = slot_lengths[i];
      for (auto j = 0; j < fea_num; ++j) {
        // Copy the emb grad from paddle tensor to BoxPS. Since
        // 'show','click','emb' are continuous in memory, so we copy here using
        // the 'show' address
        if (platform::is_cpu_place(place)) {
          memory::Copy(boost::get<platform::CPUPlace>(place),
                       reinterpret_cast<float*>(
                           &((total_grad_values_gpu + offset)->show)),
                       boost::get<platform::CPUPlace>(place),
                       grad_values[i] + j * hidden_size,
                       sizeof(float) * hidden_size);
        } else {
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
          memory::Copy(boost::get<platform::CUDAPlace>(place),
                       reinterpret_cast<float*>(
                           &((total_grad_values_gpu + offset)->show)),
                       boost::get<platform::CUDAPlace>(place),
                       grad_values[i] + j * hidden_size,
                       sizeof(float) * hidden_size, nullptr);
#endif
        }
        ++offset;
      }
    }
    */
    push_boxps_timer.Start();
    if (platform::is_cpu_place(place)) {
      int ret = boxps_ptr_->PushSparseCPU(
          reinterpret_cast<uint64_t*>(total_keys), total_grad_values_gpu,
          static_cast<int>(total_length));
      PADDLE_ENFORCE_EQ(ret, 0, "PushSparseCPU failed in BoxPS.");
    } else {
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
      VLOG(3) << "Begin call PushSparseGPU";
      int ret = boxps_ptr_->PushSparseGPU(
          reinterpret_cast<uint64_t*>(total_keys), total_grad_values_gpu,
          static_cast<int>(total_length),
          boost::get<platform::CUDAPlace>(place).GetDeviceId());
      PADDLE_ENFORCE_EQ(ret, 0, "PushSparseGPU failed in BoxPS.");
      VLOG(3) << "End call PushSparseGPU";
#endif
    }
    push_boxps_timer.Pause();
    all_timer.Pause();
    PADDLEBOX_LOG << "End PushSparseGrad in BoxWrapper: total cost: "
                  << all_timer.ElapsedSec() << " s, and push boxps cost: "
                  << push_boxps_timer.ElapsedSec() << " s";
  } else {
    PADDLE_THROW(
        "PaddleBox: PushSparse Only Support CPUPlace and CUDAPlace Now.");
  }
  VLOG(3) << "End call PushSparse";
#endif
}
}  // end namespace framework
}  // end namespace paddle