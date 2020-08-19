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
#include <fcntl.h>

#ifdef _POSIX_C_SOURCE
#undef _POSIX_C_SOURCE
#endif

#ifdef _XOPEN_SOURCE
#undef _XOPEN_SOURCE
#endif

#include <memory>
#include "paddle/fluid/framework/fleet/ascend_wrapper.h"
#include "paddle/fluid/pybind/ascend_wrapper_py.h"

namespace py = pybind11;

namespace paddle {
namespace pybind {

void BindAscendWrapper(py::module* m) {
  py::class_<framework::AscendInstance,
             std::shared_ptr<framework::AscendInstance>>(*m, "AscendInstance")
      .def(py::init([]() { return framework::AscendInstance::GetInstance(); }))
      .def("init_global_resources",
           &framework::AscendInstance::InitGlobalResouces,
           py::call_guard<py::gil_scoped_release>())
      .def("add_ascend_subgraph", &framework::AscendInstance::AddAscendSubgraph,
           py::call_guard<py::gil_scoped_release>());
}  // end AscendWrapper

}  // end namespace pybind
}  // end namespace paddle
