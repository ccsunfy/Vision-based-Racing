import onnx
import torch as th
import sys
import os
import onnxruntime as ort

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
from utils.algorithms.ppo import ppo
import numpy as np

onnx_path = "/home/suncc/SpySim6_24/sim_to_real121.onnx"
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)

# 打印ONNX模型的输入名称
print("ONNX model inputs:", [input.name for input in onnx_model.graph.input])

model = ppo.load("examples/nature_cross/ppo_436.zip", device="cuda")

# 创建测试输入
x1 = th.randn((1, 1, 64, 64))
x2 = th.randn((1, 16))
x3 = th.randn((1, 1))
x4 = th.randn((1, 1)) 
x5 = th.randn((1, 256))

# 原始模型预测
x = {
    "depth": x1,
    "state": x2,
    "vd": x3,
    "index": x4,
    "latent": x5,
}
x = {k: v.cpu().numpy() for k, v in x.items()}

with th.no_grad():
    action_orig, _ = model.policy.predict(x)
    print("Original model output:", action_orig)

# ONNX模型预测
ort_sess = ort.InferenceSession(onnx_path)

# 获取ONNX模型的输入名称
input_names = [input.name for input in ort_sess.get_inputs()]
print("ONNX Runtime input names:", input_names)

# 构造输入字典
onnx_inputs = {
    input_name: x1.numpy() if input_name == "depth" else 
                x2.numpy() if input_name == "state" else
                x3.numpy() if input_name == "vd" else
                x4.numpy() if input_name == "index" else
                x5.numpy()
    for input_name in input_names
}

# 运行ONNX推理
outputs = ort_sess.run(None, onnx_inputs)
print("ONNX model output:", outputs[0])