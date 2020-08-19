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

from paddle.fluid import program_guard, layers, default_main_program
from paddle.fluid.optimizer import Momentum, SGD
from .meta_optimizer_base import MetaOptimizerBase
from .common import OpRole, OP_ROLE_KEY, CollectiveHelper, is_update_op
import paddle.fluid.core as core


class AscendIRParser(object):
    def __init__(self):
        self.parsed_startup = None
        self.parsed_main = None

    def parse_program(self, startup_program, main_program):
        startup_subgraphs_with_id = []
        main_subgraphs_with_id = []
        # parse main program here and generate subgraph

        # A fake implementation here
        sub_graph = []
        block = startup_program.global_block()
        for i, op in list(enumerate(block.ops)):
            sub_graph.append(op.type)
        op_num = len(block.ops)
        for i in range(op_num - 1, -1, -1):
            block._remove_op(i)
        tmp_var = block.create_var(
            name="tmp", shape=[1], persistable=True, stop_gradient=True)
        block._insert_op(
            0,
            type="ascend_trigger",
            inputs={"FeedList": [tmp_var]},
            outputs={"FetchList": [tmp_var]},
            attrs={'graph_idx': 0})
        startup_subgraphs_with_id.append(sub_graph)
        sub_graph = []
        block = main_program.global_block()
        for i, op in list(enumerate(block.ops)):
            sub_graph.append(op.type)
        op_num = len(block.ops)
        for i in range(op_num - 1, -1, -1):
            block._remove_op(i)
        tmp_var = block.create_var(
            name="tmp", shape=[1], persistable=True, stop_gradient=True)
        block._insert_op(
            0,
            type="ascend_trigger",
            inputs={"FeedList": [tmp_var]},
            outputs={"FetchList": [tmp_var]},
            attrs={'graph_idx': 1})
        main_subgraphs_with_id.append(sub_graph)
        return startup_subgraphs_with_id, main_subgraphs_with_id


class AscendOptimizer(MetaOptimizerBase):
    def __init__(self, optimizer):
        super(AscendOptimizer, self).__init__(optimizer)
        self.inner_opt = optimizer
        self.meta_optimizers_white_list = []

    def _can_apply(self):
        if not self.user_defined_strategy.ascend:
            return False

        # TODO(hutuxian): other check here
        return True

    def _disable_strategy(self, dist_strategy):
        dist_strategy.ascend = False
        dist_strategy.ascend_configs = {}

    def minimize_impl(self,
                      loss,
                      startup_program=None,
                      parameter_list=None,
                      no_grad_set=None):
        minimized = self.inner_opt.minimize(
            loss, startup_program=startup_program)

        main_block = loss.block
        self.parser = AscendIRParser()
        startup_subgraphs_with_id, main_subgraphs_with_id = self.parser.parse_program(
            startup_program, main_block.program)
        self.ascend_instance = core.AscendInstance()

        self.ascend_instance.init_global_resources(
        )  # add whatever parameters here to init
        idx = 0
        for graph_with_id in startup_subgraphs_with_id:
            self.ascend_instance.add_ascend_subgraph(idx, graph_with_id)
            idx += 1
        for graph_with_id in main_subgraphs_with_id:
            self.ascend_instance.add_ascend_subgraph(idx, graph_with_id)
            idx += 1
        return minimized
