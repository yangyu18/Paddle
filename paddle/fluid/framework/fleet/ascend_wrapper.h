/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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

#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/platform/gpu_info.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/timer.h"

namespace paddle {
namespace framework {

// typedef std::vector<std::string> AscendGraphDesc;
typedef std::string AscendGraphDesc;

class AscendInstance {
 public:
  virtual ~AscendInstance() {}
  AscendInstance() {}
  // need to expose pybind function
  void InitGlobalResouces() { VLOG(0) << "InitGlobalResouces"; }

  void DestroyAscendGlobalResources() {}

  static std::shared_ptr<AscendInstance> GetInstance() {
    if (nullptr == ascend_instance_) {
      VLOG(0) << "Initialize AscendInstance";
      ascend_instance_.reset(new paddle::framework::AscendInstance());
    }
    return ascend_instance_;
  }

  void AddAscendSubgraph(int graph_idx, const AscendGraphDesc& graph) {
    ascend_graphs_.emplace_back(graph);
  }

  void RunAscendSubgraph(int graph_idx) {
    PADDLE_ENFORCE_LT(graph_idx, ascend_graphs_.size(),
                      paddle::platform::errors::PreconditionNotMet(
                          "graph_idx %d must be less than subgraph number %lu",
                          graph_idx, ascend_graphs_.size()));
    const AscendGraphDesc& graph = ascend_graphs_[graph_idx];
    // for (const auto& e : graph) {
    //   VLOG(0) << "Ascend Graph[" << graph_idx << "] run " << e;
    // }
    VLOG(0) << "Ascend Graph[" << graph_idx << "] run " << graph;
  }

 protected:
  std::vector<AscendGraphDesc> ascend_graphs_;

 private:
  static std::shared_ptr<AscendInstance> ascend_instance_;
};
}  // end namespace framework
}  // end namespace paddle
