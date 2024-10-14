# Implemenatation of GradCAM for Keras, adapted to ONNX trained with Torch backend
# See:
# https://pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/#pyis-cta-modal
# https://pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/
# https://gist.github.com/Octavio-Santana/fcd80310f05331a2843962e0ff4bb1bd
# https://plastichub.unity.cn/unity-tech-cn/ml-agents/src/commit/393630d4-f606-4735-ba20-44cd4656101b...ml-agents/mlagents/trainers
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications import imagenet_utils
from GradCAM import GradCAM
import numpy as np
import argparse
import imutils
import cv2
import onnx
from onnx2keras import onnx_to_keras

# https://pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/#pyis-cta-modal
# import the necessary packages
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import cv2
class GradCamKeras:
    def __init__(self, model, classIdx=0, layerName=None):
        # store the model, the class index used to measure the class
        # activation map, and the layer to be used when visualizing
        # the class activation map
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName
        # if the layer name is None, attempt to automatically find
        # the target output layer
        if self.layerName is None:
            self.layerName = self.find_target_layer()

        # construct our gradient model by supplying (1) the inputs
        # to our pre-trained model, (2) the output of the (presumably)
        # final 4D layer in the network, and (3) the output of the
        # softmax activations from the model
        self.gradModel = tf.keras.models.Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output,
                     self.model.output])
        #gradModel.compile(loss='categorical_crossentropy', optimizer='adam')


    def find_target_layer(self):
        # attempt to find the final convolutional layer in the network
        # by looping over the layers of the network in reverse order
        for layer in reversed(self.model.layers):
            # check to see if the layer has a 4D output
            if len(layer.output_shape) == 4 and layer.__class__.__name__ == 'Conv2D':
                return layer.name
        # otherwise, we could not find a 4D layer so the GradCAM
        # algorithm cannot be applied
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

    def compute_heatmap(self, net_input, net_output_id = 0, eps=1e-8):
        image = net_input[0]
        mask = np.ones([1, 8]).astype(np.float32)
        """
        # construct our gradient model by supplying (1) the inputs
        # to our pre-trained model, (2) the output of the (presumably)
        # final 4D layer in the network, and (3) the output of the
        # softmax activations from the model
        gradModel = tf.keras.models.Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output,
                     self.model.output])
        #gradModel.compile(loss='categorical_crossentropy', optimizer='adam')
        """

        # record operations for automatic differentiation
        with tf.GradientTape() as tape:
            # cast the image tensor to a float-32 data type, pass the
            # image through the gradient model, and grab the loss
            # associated with the specific class index
            #inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = self.gradModel(net_input)
            #loss = predictions[:, self.classIdx]

            """
            pred0 = predictions[0]
            pred1 = predictions[1]
            pred2 = predictions[2]
            pred_to_eval = pred0
            """

            pred_to_eval = predictions[net_output_id]
            #sss = tf.argmax(pred_to_eval)
            loss = pred_to_eval[:,  tf.argmax(pred_to_eval[0])]

        # use automatic differentiation to compute the gradients
        grads = tape.gradient(loss, convOutputs)
        # https://stackoverflow.com/questions/53623472/how-do-i-display-a-single-image-in-pytorch
        # tensor_image.permute(1, 2, 0)

        # compute the guided gradients
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads
        ############################
        #guidedGrads_np = guidedGrads.numpy()
        #guidedGrads_np_correct = np.zeros((1, guidedGrads_np.shape[2], guidedGrads_np.shape[3], guidedGrads_np.shape[1]))


        convOutputs_np = convOutputs
        #convOutputs_np_correct = np.zeros((1, guidedGrads_np.shape[2], guidedGrads_np.shape[3], guidedGrads_np.shape[1]))

        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]
        convOutputs = tf.transpose(convOutputs, perm=[1,2,0])
        guidedGrads = tf.transpose(guidedGrads, perm=[1, 2, 0])



        # compute the average of the gradient values, and using them
        # as weights, compute the ponderation of the filters with
        # respect to the weights
        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)


        # grab the spatial dimensions of the input image and resize
        # the output class activation map to match the input image
        # dimensions
        (w, h) = (image.shape[3], image.shape[2])
        heatmap_INTER_NEAREST = cv2.resize(cam.numpy(), (w, h), interpolation=cv2.INTER_NEAREST)
        #heatmap = cv2.resize(cam.numpy(), (w, h))
        # normalize the heatmap such that all values lie in the range
        # [0, 1], scale the resulting values to the range [0, 255],
        # and then convert to an unsigned 8-bit integer
        numer = heatmap_INTER_NEAREST - np.min(heatmap_INTER_NEAREST)
        denom = (heatmap_INTER_NEAREST.max() - heatmap_INTER_NEAREST.min()) + eps
        heatmap_INTER_NEAREST = numer / denom

        cam_numpy = cam.numpy()
        numer = cam_numpy - np.min(cam_numpy)
        denom = (cam_numpy.max() - cam_numpy.min()) + eps
        heatmap_unscaled = numer / denom

        heatmap_INTER_NEAREST_01 = heatmap_INTER_NEAREST
        heatmap_INTER_NEAREST = (heatmap_INTER_NEAREST * 255).astype("uint8")

        heatmap = cv2.resize(cam.numpy(), (w, h))
        # heatmap = cv2.resize(cam.numpy(), (w, h))
        # normalize the heatmap such that all values lie in the range
        # [0, 1], scale the resulting values to the range [0, 255],
        # and then convert to an unsigned 8-bit integer
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")
        # return the resulting heatmap to the calling function

        return [heatmap_INTER_NEAREST, heatmap_INTER_NEAREST_01, heatmap, heatmap_unscaled]

    def overlay_heatmap(self, heatmap, image, alpha=0.5,power=2,
                        colormap=cv2.COLORMAP_VIRIDIS):
                        #colormap=cv2.COLORMAP_INFERNO):
        # apply the supplied color map to the heatmap and then
        # overlay the heatmap on the input image
        heatmap_colormap = cv2.applyColorMap(heatmap, colormap)
        output = cv2.addWeighted(image, alpha, heatmap_colormap, 1 - alpha, 0)
        heatmap3d = np.zeros(image.shape)
        heatmap_help = np.power(heatmap  / 255, power)
        heatmap3d[:, :, 0] = heatmap_help
        heatmap3d[:, :, 1] = heatmap_help
        heatmap3d[:, :, 2] = heatmap_help
        output2 = image * heatmap3d
        output2 = output2.astype("uint8")
        # return a 2-tuple of the color mapped heatmap and the output,
        # overlaid image
        return (heatmap_colormap, output, output2)
