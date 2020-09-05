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


class OP(object):
    def __init__(self, varname=None):
        self.varname = varname

    def set_attr_value(self, attr_value):
        self.attr_value = attr_value
        return self

    def set_attr_index(self, attr_index):
        self.attr_index = attr_index
        return self

    def update_output_desc_y(self, tensor_desc):
        self.update_output_desc_y = tensor_desc
        return self

    def set_input_ref(self, var_init):
        return self

    def set_input_value(self, var_const):
        return self

    def set_input_x1(self, shape):
        return self

    def set_input_x2(self, shape):
        return self


def Constant():
    return OP()


def Variable(varname):
    return OP(varname)


def Assign():
    return OP()


def Data(varname):
    return OP(varname)


def Add():
    return OP()
