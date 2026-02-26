#!/usr/bin/env python

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import ModelVisionBased


args = argparse.ArgumentParser()
args.add_argument("--state_dim", type=int, default=9)
args.add_argument("--action_dim", type=int, default=9)
args.add_argument("--weight", type=str, default="exps/se3_vision/run17/checkpoint0001.pth")
args = args.parse_args()

weight = args.weight
state_dim = args.state_dim
action_dim = args.action_dim


class _M(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ModelVisionBased(state_dim, action_dim).eval()
        self.model.load_state_dict(torch.load(weight, map_location="cpu"))

    def forward(self, depth, v, h):
         # depth = F.interpolate(depth, (24, 32), mode='nearest')
        holes = depth == 0
        depth[holes] = 24.0
        depth = 3 / depth.clamp(0.3, 24) - 0.6
        # fill small holes
        # depth[holes] = F.max_pool2d(depth, (1, 3), (1, 1), (0, 1))[holes]
        depth = F.max_pool2d(depth, (2, 2))
        act, h, pred = self.model(depth, v, h)
        return act, h, pred

model = _M()

_, fake_h, prediction = model(torch.randn(1, 1, 24, 32), torch.randn(1, state_dim), None)

print(fake_h.shape)
save_path = os.path.abspath(weight)
save_path = save_path.replace(".pth", ".onnx")
torch.onnx.export(
    model,  # model being run
    (
        torch.randn(1, 1, 24, 32),
        torch.randn(1, state_dim),
        fake_h,
    ),  # model input (or a tuple for multiple inputs)
    save_path,  # where to save the model (can be a file or file-like object)
    export_params=True,  # store the trained parameter weights inside the model file
    opset_version=11,  # the ONNX version to export the model to
    do_constant_folding=True,  # whether to execute constant folding for optimization
    input_names=["img", "state", "h"],  # the model's input names
    output_names=["action", "h_out", "predicion"],  # the model's output names
)

print("export to", os.path.abspath(save_path))