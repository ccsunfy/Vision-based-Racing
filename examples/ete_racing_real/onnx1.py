#!/usr/bin/env python3

import torch as th
from typing import Tuple
import sys
import os


sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
from utils.algorithms.ppo import ppo
from utils.type import TensorDict

# class OnnxablePolicy2(th.nn.Module):
#     def __init__(self, policy):
#         super().__init__()
#         self.policy = policy

#     def forward(self, x1, x2, x3, x4, x5) -> Tuple[th.Tensor, th.Tensor]:
#         # observation = {
#         #     "depth": observation_tuple[0],
#         #     "state": observation_tuple[1],
#         #     "vd": observation_tuple[2],
#         #     "index": observation_tuple[3],
#         #     "latent": observation_tuple[4]
#         # }
#         # x1, x2, x3, x4, x5 = x
#         # x = [x1.numpy(), x2.numpy(), x3.numpy(), x4.numpy(), x5.numpy()]
        
#         x = {
#                 "depth": x1,
#                 "state": x2,
#                 "vd": x3,
#                 "index": x4,
#                 "latent": x5,
#                 }
#         x = {k: v.cpu().numpy() for k, v in x.items()}
        
#         # observation = {
#         #     "depth": observation_tuple[0].contiguous(),
#         #     "state": observation_tuple[1].contiguous(),
#         #     "vd": observation_tuple[2].contiguous(),
#         #     "index": observation_tuple[3].contiguous(),
#         #     "latent": observation_tuple[4].contiguous()
#         # }
#         with th.no_grad():
#             action, _,h = self.policy.predict(x)
#             action = th.from_numpy(action)
#         # return action, value
#         return action,h, x1, x2, x3, x4, x5



# model = ppo.load("examples/ete_racing_real/demo2_onboard_empty_719_4.zip", device="cuda")
# model = ppo.load("examples/ete_racing_real/demo3_empty_728_2.zip", device="cuda")
model = ppo.load("examples/ete_racing_real/demo3_empty_829_with_1_depth_1.zip", device="cuda")
# onnxable_model = OnnxablePolicy2(model.policy)
onnxable_model = model.policy
# print(onnxable_model)
# observation = {
#         "depth": th.randn(1, 1, 64, 64),
#         "state": th.randn(1, 16),
#         "vd": th.randn(1,1),
#         "index": th.randn(1,1),
#         "latent": th.randn(1,256),
#         }
# obs = {k: v.cpu().numpy() for k, v in observation.items()}
# observation = {
#     "depth": th.randn(1, 1, 64, 64),
#     "state": th.randn(1, 16),
#     "vd": th.randn(1, 1),
#     "index": th.randn(1, 1),
#     "latent": th.randn(1, 256)
# }
th.manual_seed(42)
x1 = th.randn((1,1, 64, 64)).cuda()
x2 = th.randn((1,16)).cuda()
x3 = th.randn((1,1)).cuda()
x4 = th.randn((1,1)).cuda() 
# x5 = th.randn((1,256)).cuda()
# x = (x1, x2, x3, x4)

# observation_tuple = (
#     observation["depth"],
#     observation["state"],
#     observation["vd"],
#     observation["index"],
#     observation["latent"],
# )
# 构建元组
# observation_tuple = tuple(observation.values())

onnxable_model.eval()
# action = onnxable_model(x1, x2, x3, x4, x5)
# print(x1.shape, x2.shape, x3.shape, x4.shape, x5.shape)
th.onnx.export(
    onnxable_model,
    # (x1, x2, x3, x4),
    # args=(x1, x2, x3, x4, x5)3
    args=(x1,x2,x3,x4),
    # f="demo2_onboard_empty_719.onnx",
    f="demo3_empty_829_with_1_depth_1_rate_3.onnx",
    opset_version=17,
    export_params=True, # store the trained parameter weights inside the model file
    do_constant_folding=True, # whether to execute constant folding for optimization
    input_names = ['depth', 'state', 'vd','index'],
    # input_names = ['state', 'vd', 'index','latent'],
    # input_names = ['depth','state','index'],
    # output_names = ['action', 'h', 'depth1', 'state1', 'vd1', 'index1', 'latent1'],
    output_names = ['actions', 'values', 'log_prob']
    # output_names = ['actions', 'values', 'log_prob','h']
    #  output_names = ['actions', 'values','h']
    # verbose=True,
    # output_names = ['action', 'latent']
)
