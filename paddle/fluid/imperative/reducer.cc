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

void Reducer::Print_Data() {
  std::cout << "Currently, we can set the number :";
}

std::vector<std::vector<size_t>> assign_bucket_by_size(
    const std::vector<std::shared_ptr<imperative::VarBase>> &vars,
    const std::vector<size_t> &bucket_size_limits) {
  std::cout << "vars.size()" << vars.size() << std::endl;

  // the return vectorr
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
          std::min(cur_limit_index + 1, bucket_size_limits.size());
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
