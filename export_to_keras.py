# Export ONNX model to Keras
# https://github.com/gmalivenko/onnx2keras
import onnx
from onnx2keras import onnx_to_keras

# https://github.com/onnx/onnx/issues/2052
# Load ONNX model
onnx_model = onnx.load("brains/nature56x56_hidden16/modified_SimpleCollector_56x56_hidden16.onnx")
img_size = 56
# Call the converter (input - is the main model input name, can be different for your model)
k_model = onnx_to_keras(onnx_model, ['obs_0','action_masks'], name_policy='renumerate')

import numpy as np
my_data = np.random.rand(1,3,img_size,img_size).astype(np.float32)
my_data = np.multiply(my_data, 255.0)
import cv2
img = cv2.imread('c:/data/Camera/0000000000.png')
img = cv2.resize(img, [img_size, img_size])
img_respahed = np.reshape(img, [1,3,img_size,img_size]).astype(np.float32)
mask = np.ones([1,8]).astype(np.float32)

res = k_model([img_respahed, mask])
print(res)
k_model.save('brains/nature56x56_hidden16/my_model_nature')
k_model.summary()
