import onnx
import onnxruntime as ort
import numpy as np
import torch as th
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
from utils.algorithms.ppo import ppo

onnx_path = "demo1_onboard_610.onnx"
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
model = ppo.load("examples/ete_racing_real/demo1_onboard_610.zip", device="cpu")

ort_sess = ort.InferenceSession(onnx_path)

input_names = [input.name for input in ort_sess.get_inputs()]
print("ONNX模型输入名称:", input_names)
output_names = [output.name for output in ort_sess.get_outputs()]
print("ONNX模型输出名称:", output_names)

th.random.manual_seed(42)
x1 = th.randn((1,1, 64, 64,), dtype=th.float32)
x2 = 1*th.randn((1,16,), dtype=th.float32)
x3 = th.randn((1,1,), dtype=th.float32)
# x4 = th.randn((1,1,), dtype=th.float32)
x5 = th.randn((1,256,), dtype=th.float32)
    
# hide_state = x5.numpy()
# inputs = {
# 'depth': x1.numpy(),
# 'state': x2.numpy(),
# 'vd': x3.numpy(),
# 'index': x4.numpy(),
# 'latent': hide_state
# }
inputs = {
'depth': x1.numpy(),
'state': x2.numpy(),
'vd': x3.numpy(),
# 'index': x4.numpy(),
'latent': x5.numpy() 
}


onnx_actions = ort_sess.run(None, inputs)
onnx_action = onnx_actions[0]
# onnx_value = onnx_actions[1]
# onnx_h = onnx_actions[2]
onnx_action = np.clip(onnx_action, -1,1)  # type: ignore[assignment, arg-type]
# print("onnx_action_mean ",onnx_action_mean)
# mean_actions = th.from_numpy(onnx_action_mean)

# value = th.from_numpy(onnx_value)
# h = th.from_numpy(onnx_h)
# action,state,h = model.policy.postprec(mean_actions,value,h)

print("onnx-postprec ",onnx_action)

# Check that the predictions are the same
with th.no_grad():
    actions, _, h = model.policy.predict(inputs, deterministic=True) # use GRU
    # actions, _ = model.policy.predict(inputs, deterministic=True) # no GRU

    
    # mean_actions, values, h = model.policy.forward1(x1,x2,x3,x4,x5)
    # action,state,h = model.policy.postprec(mean_actions,value,h)
    
    print("torch action",actions)
    
