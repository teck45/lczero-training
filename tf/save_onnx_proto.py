import proto.net_pb2 as pb
from net import Net

"""
This file loads an onnx model using net.py, modifies the names of the 
input and output layers to match the naming in tfprocess, and saves 
it as a proto file that can be read and loaded by the onnx backend.

This file is step 9 in the conversion process.

Script provided by Borg.
"""

net = Net(net=pb.NetworkFormat.NETWORK_ONNX)
file = open("/path/to/tf2onnx_models/model.onnx", "rb")
model = file.read()
file.close()
net.pb.onnx_model.model = model
net.pb.onnx_model.input_planes = "input_1"
net.pb.onnx_model.output_wdl = "value/dense2"
net.pb.onnx_model.output_policy = "apply_attention_policy_map"  # "apply_policy_map" if converting a standard net
net.pb.onnx_model.output_mlh = "moves_left/dense2"
net.save_proto("/path/to/lc0/lc0networks/model_name")
