/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <glog/logging.h>
#include <atomic>
#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <unordered_set>
#include <vector>
#include "paddle/fluid/framework/data_set.h"
#include "paddle/fluid/platform/timer.h"
#ifdef PADDLE_WITH_BOX_PS
#include <boxps_public.h>
#endif
#include "paddle/fluid/platform/gpu_info.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace framework {

#define PADDLEBOX_LOG VLOG(0) << "PaddleBox: "

class BoxWrapper {
 public:
  virtual ~BoxWrapper() {}
  BoxWrapper() {}

  void FeedPass(const std::vector<uint64_t>& feasgin_to_box) const;
  void BeginPass() const;
  void EndPass() const;
  void ResetClickNum();
  void UpdateClickNum(int64_t actual_val, float pred_val);
  void PrintClickNum() const;
  void PullSparse(const paddle::platform::Place& place,
                  const std::vector<const uint64_t*>& keys,
                  const std::vector<float*>& values,
                  const std::vector<int64_t>& slot_lengths,
                  const int hidden_size);
  void PushSparseGrad(const paddle::platform::Place& place,
                      const std::vector<const uint64_t*>& keys,
                      const std::vector<const float*>& grad_values,
                      const std::vector<int64_t>& slot_lengths,
                      const int hidden_size);
  void InitializeGPU(const char* conf_file) {
    if (nullptr != s_instance_) {
      PADDLEBOX_LOG << "Begin InitializeGPU";
#ifdef PADDLE_WITH_BOX_PS
      std::vector<cudaStream_t*> stream_list;
      for (int i = 0; i < platform::GetCUDADeviceCount(); ++i) {
        VLOG(3) << "before get context i[" << i << "]";
        platform::CUDADeviceContext* context =
            dynamic_cast<platform::CUDADeviceContext*>(
                platform::DeviceContextPool::Instance().Get(
                    platform::CUDAPlace(i)));
        PADDLEBOX_LOG << "after get cuda context for card [" << i << "]";
        PADDLE_ENFORCE_EQ(context == nullptr, false, "context is nullptr");

        stream_list_[i] = context->stream();
        stream_list.push_back(&stream_list_[i]);
      }
      PADDLEBOX_LOG << "call InitializeGPU in boxps";
      // the second parameter is useless
      s_instance_->boxps_ptr_->InitializeGPU(conf_file, -1, stream_list);
      PADDLEBOX_LOG << "return from InitializeGPU in boxps";
#endif
    }
  }
  void Finalize() {
    if (nullptr != s_instance_) {
      s_instance_->boxps_ptr_->Finalize();
    }
  }
  void SaveModel() const { printf("savemodel in box_Wrapper\n"); }
  void LoadModel() const { printf("loadmodel in box\n"); }

  static std::shared_ptr<BoxWrapper> GetInstance() {
    if (nullptr == s_instance_) {
      // If main thread is guaranteed to init this, this lock can be removed
      static std::mutex mutex;
      std::lock_guard<std::mutex> lock(mutex);
      if (nullptr == s_instance_) {
        VLOG(3) << "s_instance_ is null";
        s_instance_.reset(new paddle::framework::BoxWrapper());
#ifdef PADDLE_WITH_BOX_PS
        s_instance_->boxps_ptr_.reset(boxps::BoxPSBase::GetIns());
#endif
      }
    }
    return s_instance_;
  }

 private:
#ifdef PADDLE_WITH_BOX_PS
  static cudaStream_t stream_list_[8];
  static std::shared_ptr<boxps::BoxPSBase> boxps_ptr_;
#endif
  static std::shared_ptr<BoxWrapper> s_instance_;
  int GetDate() const;
  // will be failed when multi datasets run concurrently.
  std::atomic<int64_t> actual_click;
  float pred_click;
  std::mutex add_mutex;
};

class BoxHelper {
 public:
  explicit BoxHelper(paddle::framework::Dataset* dataset) : dataset_(dataset) {}
  virtual ~BoxHelper() {}

  void BeginPass() {
    auto box_ptr = BoxWrapper::GetInstance();
    box_ptr->BeginPass();
  }

  void EndPass() {
    auto box_ptr = BoxWrapper::GetInstance();
    box_ptr->EndPass();
  }
  void LoadIntoMemory() {
    platform::Timer timer;
    PADDLEBOX_LOG << "Begin LoadIntoMemory(), dataset[" << dataset_ << "]";
    timer.Start();
    dataset_->LoadIntoMemory();
    timer.Pause();
    PADDLEBOX_LOG << "download + parse cost: " << timer.ElapsedSec() << "s";

    timer.Start();
    FeedPass();
    timer.Pause();
    PADDLEBOX_LOG << "FeedPass cost: " << timer.ElapsedSec() << " s";
    PADDLEBOX_LOG << "End LoadIntoMemory(), dataset[" << dataset_ << "]";
  }
  void PreLoadIntoMemory() {
    dataset_->PreLoadIntoMemory();
    feed_data_thread_.reset(new std::thread([&]() {
      dataset_->WaitPreLoadDone();
      FeedPass();
    }));
    VLOG(3) << "After PreLoadIntoMemory()";
  }
  void WaitFeedPassDone() { feed_data_thread_->join(); }

 private:
  Dataset* dataset_;
  std::shared_ptr<std::thread> feed_data_thread_;
  // notify boxps to feed this pass feasigns from SSD to memory
  void FeedPass() {
    auto box_ptr = BoxWrapper::GetInstance();
    auto input_channel_ =
        dynamic_cast<MultiSlotDataset*>(dataset_)->GetInputChannel();
    std::vector<Record> pass_data;
    std::vector<uint64_t> feasign_to_box;
    std::unordered_set<uint64_t> feasign_to_box_set;
    input_channel_->ReadAll(pass_data);

    auto& index_map = dataset_->GetReaders()[0]->index_omited_in_feedpass_;
    for (const auto& ins : pass_data) {
      const auto& feasign_v = ins.uint64_feasigns_;
      for (const auto feasign : feasign_v) {
        if (index_map.find(feasign.slot()) != index_map.end()) {
          continue;
        }
        feasign_to_box_set.insert(feasign.sign().uint64_feasign_);
      }
    }
    feasign_to_box.assign(feasign_to_box_set.begin(), feasign_to_box_set.end());
    input_channel_->Open();
    input_channel_->Write(pass_data);
    input_channel_->Close();
    PADDLEBOX_LOG << "call boxps feedpass";
    box_ptr->FeedPass(feasign_to_box);
    PADDLEBOX_LOG << "return from boxps feedpass";
  }
};

}  // end namespace framework
}  // end namespace paddle
