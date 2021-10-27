Convert nets to be compatible with the onnx backend using the following workflow:

in train.py:
1. load the last checkpoint from training (or create one first with `net_to_model.py` when training checkpoints are absent, like with net 744204 provided)
2. load the SWA weights using code from `tfprocess.py`
3. save the network in TensorFlow 'saved model' format using model.save(path)

in fix_tf_model.py:
4. load the tf model configuration using model.get_config()
5. directly modify the model config to resolve net arch inconsistencies between lc0 backend and training code, specifically:
    * change input layer shape to [112x8x8] from [112x64]
    * remove reshape layer after input and replace it with a TFOpLambda layer to scale the rule 50 plane by 1/99, because the onnx backend expects this
    * resolve the necessary 'inbound_nodes' fields after insertion of new layer
    * add softmax activation to the value head output
6. create a new model from the modified config using `model.__class__.from_config(model_config, custom_objects)`, passing it the definitions of the custom object layers from tfprocess: ApplySqueezeExcitation and ApplyPolicyMap / ApplyAttentionPolicyMap 
7. transfer over the layer weights to the new model and save it

from the command line:
8. convert the saved model to onnx using tf2onnx (https://github.com/onnx/tensorflow-onnx/releases)

in save_onnx_proto.py:
9. load the onnx model, resolve the naming of input and output layers, and save the weights as a proto file
