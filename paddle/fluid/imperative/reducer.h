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

#pragma once

#include <memory>
#include <vector>
#include "paddle/fluid/imperative/layer.h"

namespace paddle {
namespace imperative {

struct Bucket {
  framework::Tensor contents_;
  std::vector<size_t> offset_;
  std::vector<framework::Variable> vars_;

  // Global indices of participating variables in the bucket
  std::vector<size_t> variable_indices;

  // Number of params that haven't been ready
  size_t pending = -1;
  bool is_sparse_ = false;
};

class Reducer {
 public:
  Reducer() {}

  virtual ~Reducer() {}

  void Print_Data();

  // Reducer singleton
  static std::shared_ptr<Reducer> GetInstance() {
    if (NULL == s_instance_) {
      s_instance_.reset(new paddle::imperative::Reducer());
    }
    return s_instance_;
  }

 protected:
 private:
  static std::shared_ptr<Reducer> s_instance_;
};

std::vector<std::vector<size_t>> assign_bucket_by_size(
    const std::vector<std::shared_ptr<imperative::VarBase>>& tensors,
    const std::vector<size_t>& bucket_size_limits);

}  // namespace imperative
}  // namespace paddle
