# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FORMAT_ND = "FORMAT_ND"
DT_FLOAT16 = "DT_FLOAT16"


class Shape(object):
    def __init__(self, shape):
        self.shape = shape

    def get_dim_num(self):
        return len(self.shape)

    def get_dims(self):
        return self.shape


class TensorDesc(object):
    def __init__(self, shape, format, dtype):
        self.shape = shape
        self.format = format
        self.type = dtype

    def get_shape(self):
        return self.shape

    def set_real_dim_cnt(self, real_dim_cnt):
        self.real_dim_cnt = real_dim_cnt
        print("set_real_dim_cnt: ", real_dim_cnt)


class Tensor(object):
    def __init__(self):
        pass

    def set_tensor_desc(self, input_tensor_desc):
        self.input_tensor_desc = input_tensor_desc

    def set_data(self, data):
        self.data = data


class OP(object):
    def __init__(self):
        pass

    def set_attr_value(tensor):
        self.tensor = tensor


class Graph(object):
    def __init__(self, name):
        self.name = name
        self.oplist = []

    def set_inputs(self, operator):
        self.inputs = operator
        return self

    def set_outputs(self, operator):
        self.outputs = operator
        return self

    def add_op(self, op):
        self.oplist.append(op)

    def __str__(self):
        ret = "Ascend Subgraph %s has %d ops" % (self.name, len(self.oplist))
        return ret
