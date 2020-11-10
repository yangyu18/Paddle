// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/imperative/reducer.h"
#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include "paddle/fluid/framework/data_type.h"

namespace paddle {
namespace imperative {

std::shared_ptr<Reducer> Reducer::s_instance_ = NULL;

Reducer::Reducer(const std::vector<std::shared_ptr<imperative::VarBase>> &vars,
                 const std::vector<std::vector<size_t>> &bucket_indices)
    : vars_(vars), bucket_indices_(bucket_indices) {
  VLOG(0) << "Start construct the Reducer ...";
  // initialize buckets
  initialize_buckets(bucket_indices);

  // initialize varname2index_
  {
    for (size_t bucket_index = 0; bucket_index < bucket_indices.size();
         ++bucket_index) {
      for (size_t var_index = 0;
           var_index < bucket_indices[bucket_index].size(); ++var_index) {
        size_t index = bucket_indices[bucket_index][var_index];
        const std::string &var_name = vars_[index]->GradVarName();
        varname2index_[var_name] = VariableIndex{
            .bucket_index = bucket_index, .variable_index = var_index,
        };
      }
    }
  }

  // initialize DeviceContext
  int device_id = BOOST_GET_CONST(platform::CUDAPlace, place_).GetDeviceId();
  std::unique_ptr<paddle::platform::CUDADeviceContext> dev_ctx(
      new paddle::platform::CUDADeviceContext(platform::CUDAPlace(device_id)));
  dev_ctx_ = std::move(dev_ctx);

  // release DeviceContext
  std::call_once(once_flag_, []() {
    std::atexit([]() { Reducer::GetInstance()->ReleaseDevCtx(); });
  });
}

void Reducer::initialize_buckets(
    const std::vector<std::vector<size_t>> &bucket_indices) {
  VLOG(0) << "Start initialize buckets ..";
  // clear the bucket
  buckets_.clear();
  buckets_.reserve(bucket_indices.size());

  auto bucket_nums = bucket_indices.size();
  for (size_t bucket_index = 0; bucket_index < bucket_nums; ++bucket_index) {
    Bucket bucket;
    int64_t all_length = 0;
    // bucket.pending = bucket_indices[bucket_index].size();
    bucket.variable_indices_ = bucket_indices[bucket_index];
    size_t offset = 0;

    for (size_t index = 0; index < bucket_indices[bucket_index].size();
         ++index) {
      const auto variable_index = bucket_indices[bucket_index][index];
      const auto &var = vars_[variable_index];
      const auto var_name = var->Name();

      // TODO(shenliang03): to process the selectrows
      auto lod_tensor = var->MutableVar()->GetMutable<framework::LoDTensor>();

      PADDLE_ENFORCE_EQ(lod_tensor->IsInitialized(), true,
                        platform::errors::InvalidArgument(
                            "Tensor `%s` is not initialized.", var_name));
      auto size = lod_tensor->numel();
      PADDLE_ENFORCE_GT(
          size, 0, platform::errors::InvalidArgument(
                       "The number of tensor `%s`'s elements is 0.", var_name));
      all_length += size;

      bucket.offset_.push_back(offset);
      bucket.length_.push_back(size);
      offset += size;

      // check the dtype and place, it must be same.
      auto dtype = var->DataType();
      auto place = var->Place();
      if (index > 0) {
        PADDLE_ENFORCE_EQ(dtype, bucket.dtype,
                          platform::errors::InvalidArgument(
                              "Tensor `%s` has different dtype.", var_name));
        PADDLE_ENFORCE_EQ(place, place_,
                          platform::errors::InvalidArgument(
                              "Tensor `%s` has different place.", var_name));
      } else {
        bucket.dtype = dtype;
        place_ = place;
      }
    }

    // Alloc the continuous space
    bucket.contents.Resize(framework::make_ddim({all_length}))
        .mutable_data(place_, bucket.dtype);

    // Debug Message For Reducer
    VLOG(0) << "the buckets_[" << bucket_index << "] basic message:";
    VLOG(0) << "all_length" << all_length;
    VLOG(0) << "offset:";
    for (auto ele : bucket.offset_) VLOG(0) << ele;
    VLOG(0) << "length:";
    for (auto ele : bucket.length_) VLOG(0) << ele;

    buckets_.emplace_back(std::move(bucket));
  }
}

void Reducer::set_grad_space(const Bucket &bucket) {
  const std::vector<size_t> &global_indices = bucket.variable_indices_;
  const auto &offset = bucket.offset_;
  const auto &length = bucket.length_;
  for (size_t index = 0; index < global_indices.size(); ++index) {
    const auto &var = vars_[global_indices[index]];  // varbase of var
    auto &grad_var = var->GradVarBase();             // varbase of var grad
    auto grad_tensor =
        grad_var->MutableVar()->GetMutable<framework::LoDTensor>();
    auto dim = grad_tensor->dims();
    grad_tensor
        ->ShareDataWith(bucket.contents.Slice(
            static_cast<int64_t>(offset[index]),
            static_cast<int64_t>(offset[index] + length[index])))
        .Resize(dim);
  }
}

void Reducer::prepare_for_backward() {
  VLOG(0) << "start reseting count..";
  next_bucket_ = 0;
  for (size_t bucket_index = 0; bucket_index < buckets_.size();
       ++bucket_index) {
    auto &bucket = buckets_[bucket_index];
    bucket.pending = bucket.variable_indices_.size();
  }
}

void Reducer::add_dist_hook(VariableWrapper *var_warpper) {
  const std::string &var_name = var_warpper->Name();
  // VLOG(0) << "before add_dist_hook, varname is " << var_name;
  if (varname2index_.find(var_name) == varname2index_.end()) {
    VLOG(3) << "This " << var_name << " is not trainable";
    return;
  }

  VariableIndex var_index = varname2index_[var_name];

  auto bucket_index = var_index.bucket_index;
  auto &bucket = buckets_[bucket_index];

  mark_variable_ready(var_index, var_warpper);
  if (--bucket.pending == 0) {
    // can start allreduce
    mark_bucket_ready(bucket_index);
  }

  if (next_bucket_ == buckets_.size()) {
    finalize_backward();
  }
}

void Reducer::mark_variable_ready(const VariableIndex &var_index,
                                  VariableWrapper *var_warpper) {
  auto bucket_index = var_index.bucket_index;
  auto variable_index = var_index.variable_index;
  auto &bucket = buckets_[bucket_index];
  auto offset = bucket.offset_[variable_index];
  auto length = bucket.length_[variable_index];
  auto &contents = bucket.contents;

  auto tensor = var_warpper->MutableVar()->GetMutable<framework::LoDTensor>();
  const auto &var_dtype = var_warpper->DataType();
  void *src_data = tensor->mutable_data(var_warpper->Place(), var_dtype);

  void *dst_data = contents
                       .Slice(static_cast<int64_t>(offset),
                              static_cast<int64_t>(offset + length))
                       .mutable_data(place_, bucket.dtype);
  // use cal_stream
  auto *cal_stream = static_cast<platform::CUDADeviceContext *>(
                         platform::DeviceContextPool::Instance().Get(place_))
                         ->stream();
  memory::Copy(BOOST_GET_CONST(platform::CUDAPlace, place_), dst_data,
               BOOST_GET_CONST(platform::CUDAPlace, var_warpper->Place()),
               src_data, framework::SizeOfType(var_dtype) * length, cal_stream);
}

void Reducer::mark_bucket_ready(size_t bucket_index) {
  if (bucket_index > next_bucket_) return;
  for (; next_bucket_ < buckets_.size() && buckets_[next_bucket_].pending == 0;
       ++next_bucket_) {
    SyncCalStream(place_);
    // Debug Message
    // VLOG(0) << "next_bucket_ : " << next_bucket_;
    // VLOG(0) << "dims: " << buckets_[next_bucket_].contents.dims();
    AllReduce(buckets_[next_bucket_].contents,
              &(buckets_[next_bucket_].contents), dev_ctx_);
  }
}

void Reducer::finalize_backward() {
  SyncCommStream(dev_ctx_);

  VLOG(0) << "set gradient space by bucket";
  for (auto &bucket : buckets_) {
    set_grad_space(bucket);
  }
  VLOG(3) << "finalize_backward is finished...";
}

void Reducer::ReleaseDevCtx() { dev_ctx_.reset(); }

std::vector<std::vector<size_t>> assign_bucket_by_size(
    const std::vector<std::shared_ptr<imperative::VarBase>> &vars,
    const std::vector<size_t> &bucket_size_limits) {
  // the return vector
  std::vector<std::vector<size_t>> res;

  // Key: the var type
  // Value: should use which index in bucket_size_limits for bucket size limit
  std::unordered_map<std::string, size_t> bucket_limit_index;

  // Key: the var type
  // Value: <the var index in input tensors, total numel in this bucket>
  std::unordered_map<std::string, std::pair<std::vector<size_t>, size_t>>
      next_bucket;

  for (size_t i = 0; i < vars.size(); ++i) {
    const auto &var = vars[i];
    if (var->Var().IsType<framework::SelectedRows>()) {
      // we keep sparse var a single bucket
      res.push_back({i});
      continue;
    }

    const auto &var_dtype = var->DataType();
    const auto var_dtype_str = framework::DataTypeToString(var_dtype);
    VLOG(0) << "var[" << var->GradVarName() << "] 's type is "
            << var->DataType();
    auto &bucket_info = next_bucket[var_dtype_str];
    int64_t var_size = -1;
    if (var->Var().IsType<framework::LoDTensor>()) {
      var_size = var->Var().Get<framework::LoDTensor>().numel();
      VLOG(0) << "dims: " << var->Var().Get<framework::LoDTensor>().dims();
    } else {
      VLOG(0) << "var " << var->Name()
              << " is not tensor or selected_rows, so skip it";
      continue;
    }
    VLOG(0) << "var[" << var->GradVarName() << "] 's size is " << var_size;
    bucket_info.first.push_back(i);
    bucket_info.second += framework::SizeOfType(var_dtype) * var_size;

    if (bucket_limit_index.find(var_dtype_str) == bucket_limit_index.end()) {
      // means it is the first var of var_dtype
      bucket_limit_index[var_dtype_str] = 0;
    }
    auto &cur_limit_index = bucket_limit_index[var_dtype_str];
    if (bucket_info.second >= bucket_size_limits[cur_limit_index]) {
      // exceed bucket capacity and create a new bucket
      res.emplace_back(std::move(bucket_info.first));
      bucket_info = std::pair<std::vector<size_t>, size_t>();
      cur_limit_index =
          std::min(cur_limit_index + 1, bucket_size_limits.size() - 1);
    }
  }

  // add the final buckets
  for (auto &e : next_bucket) {
    auto &bucket_info = e.second;
    if (!bucket_info.first.empty()) {
      res.emplace_back(std::move(bucket_info.first));
    }
  }

  for (const auto &bucket_index : res) {
    PADDLE_ENFORCE_NE(
        bucket_index.empty(), true,
        platform::errors::PreconditionNotMet(
            "assign_bucket_by_size construct empty bucket, please check"));
  }
  std::sort(res.begin(), res.end(),
            [](const std::vector<size_t> &x, const std::vector<size_t> &y) {
              return x.front() < y.front();
            });
  return res;
}

}  // namespace imperative
}  // namespace paddle
