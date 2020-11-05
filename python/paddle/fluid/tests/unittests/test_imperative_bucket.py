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

from __future__ import print_function

import contextlib
import unittest
import numpy as np
import six
import unittest

import paddle
import paddle.fluid as fluid
import paddle.fluid.dygraph as dygraph
from paddle.fluid.dygraph.nn import Linear
import paddle.fluid.core as core


class MLP(fluid.Layer):
    def __init__(self, param_attr=None, bias_attr=None):
        super(MLP, self).__init__()

        self._linear1 = Linear(784, 10)
        self._linear2 = Linear(10, 10)

    def forward(self, inputs):
        y = self._linear1(inputs)
        y = self._linear2(y)
        return y


class TestDataParallelBucket(unittest.TestCase):
    def create_varbase(self, dtype, shape):
        return core.VarBase(dtype, shape, "", core.VarDesc.VarType.LOD_TENSOR,
                            True)

    def test_construct_bucket1(self):
        # one dtype & one limit cap
        var_list = []
        var_list.append(self.create_varbase(core.VarDesc.VarType.FP32, [2, 50]))
        var_list.append(
            self.create_varbase(core.VarDesc.VarType.FP32, [2, 100]))
        var_list.append(self.create_varbase(core.VarDesc.VarType.FP32, [2, 50]))
        var_list.append(self.create_varbase(core.VarDesc.VarType.FP32, [2, 25]))
        res = core.assign_bucket_by_size(var_list, [400])
        self.assertEqual([[0], [1], [2], [3]], res)

    def test_construct_bucket1(self):
        # multi dtype & one limit cap
        var_list = []
        var_list.append(self.create_varbase(core.VarDesc.VarType.FP32, [1, 50]))
        var_list.append(self.create_varbase(core.VarDesc.VarType.FP64, [1, 25]))
        var_list.append(self.create_varbase(core.VarDesc.VarType.FP32, [1, 50]))
        var_list.append(self.create_varbase(core.VarDesc.VarType.FP64, [1, 25]))
        var_list.append(self.create_varbase(core.VarDesc.VarType.FP32, [1, 50]))
        var_list.append(self.create_varbase(core.VarDesc.VarType.FP64, [1, 25]))
        res = core.assign_bucket_by_size(var_list, [400])
        self.assertEqual([[0, 2], [1, 3], [4], [5]], res)

    def test_construct_bucket2(self):
        # one dtype & multi limit cap
        var_list = []
        var_list.append(self.create_varbase(core.VarDesc.VarType.FP32, [2, 50]))
        var_list.append(self.create_varbase(core.VarDesc.VarType.FP32, [2, 50]))
        var_list.append(self.create_varbase(core.VarDesc.VarType.FP32, [2, 50]))
        var_list.append(self.create_varbase(core.VarDesc.VarType.FP32, [2, 50]))
        res = core.assign_bucket_by_size(var_list, [400, 800])
        self.assertEqual([[0], [1, 2], [3]], res)

    def test_construct_bucket3(self):
        # multi dtype & multi limit cap
        var_list = []
        var_list.append(self.create_varbase(core.VarDesc.VarType.FP32, [1, 50]))
        var_list.append(self.create_varbase(core.VarDesc.VarType.FP64, [1, 25]))
        var_list.append(self.create_varbase(core.VarDesc.VarType.FP32, [1, 50]))
        var_list.append(self.create_varbase(core.VarDesc.VarType.FP64, [1, 25]))
        var_list.append(self.create_varbase(core.VarDesc.VarType.FP32, [1, 50]))
        var_list.append(self.create_varbase(core.VarDesc.VarType.FP64, [1, 25]))
        res = core.assign_bucket_by_size(var_list, [200, 400])
        self.assertEqual([[0], [1], [2, 4], [3, 5]], res)


class TestDataParallelStateDict(unittest.TestCase):
    def test_data_parallel_state_dict(self):
        with fluid.dygraph.guard():
            strategy = paddle.distributed.prepare_context()
            mlp = MLP()
            parallel_mlp = dygraph.parallel.DataParallel(mlp, strategy)

            data_numpy = np.random.random([1, 784])
            lablel_numpy = np.random.randint(1, 5, [10, 1])
            data_numpy = data_numpy.astype("float32")
            lablel_numpy = lablel_numpy.astype("float32")
            img = paddle.to_tensor(data_numpy)
            label = paddle.to_tensor(lablel_numpy)

            out = parallel_mlp(img)
            mse_loss = paddle.nn.loss.MSELoss()
            loss = mse_loss(input=out, label=label)
            loss.backward()


if __name__ == '__main__':
    unittest.main()
