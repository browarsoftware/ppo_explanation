# Dsiplay GradCAM of Keras model and save results to file
from GradCamKeras import GradCamKeras
import imutils
import tensorflow as tf
import numpy as np

def prepareImage(img, imgage_dim):
    img_reshaped = np.zeros([1,3,imgage_dim,imgage_dim])
    for a in range(imgage_dim):
        for b in range(imgage_dim):
            for c in range(3):
                img_reshaped[0, c, a, b] = img[a, b, 2 - c]
    img_reshaped = np.multiply(img_reshaped, 1/255.).astype(np.float32)
    return img_reshaped

def returnPredictions(preds):
    preds_argmax = []
    for p in preds:
        preds_argmax.append(np.argmax(p))
    return preds_argmax

#model = tf.keras.models.load_model('C:/projects/python/pythonProjectOnnxToTensorflow/brains/nature192x192_hidden16/my_model_nature') # 20x20
#model = tf.keras.models.load_model('C:/projects/python/pythonProjectOnnxToTensorflow/brains/nature128x128_hidden16/my_model_nature') # 12x12
#model = tf.keras.models.load_model('C:/projects/python/pythonProjectOnnxToTensorflow/brains/nature96x96_hidden16/my_model_nature') # 8x8
#model = tf.keras.models.load_model('C:/projects/python/pythonProjectOnnxToTensorflow/brains/nature64x64_hidden16/my_model_nature') # 4x4
#model = tf.keras.models.load_model('C:/projects/python/pythonProjectOnnxToTensorflow/brains/nature56x56_hidden16/my_model_nature') # 4x4
#model = tf.keras.models.load_model('C:/projects/python/pythonProjectOnnxToTensorflow/brains/nature48x48_hidden16/my_model_nature') # 2x2

#model = tf.keras.models.load_model('C:/projects/python/pythonProjectOnnxToTensorflow/brains/simple192x192_hidden16/my_model_simple') # 22x22
#model = tf.keras.models.load_model('C:/projects/python/pythonProjectOnnxToTensorflow/brains/simple128x128_hidden16/my_model_simple') # 14x14
#model = tf.keras.models.load_model('C:/projects/python/pythonProjectOnnxToTensorflow/brains/simple96x96_hidden16/my_model_simple') # 10x10
#model = tf.keras.models.load_model('C:/projects/python/pythonProjectOnnxToTensorflow/brains/simple64x64_hidden16/my_model_simple') # 6x6
#model = tf.keras.models.load_model('C:/projects/python/pythonProjectOnnxToTensorflow/brains/simple56x56_hidden16/my_model_simple') # 5x5
model = tf.keras.models.load_model('C:/projects/python/pythonProjectOnnxToTensorflow/brains/simple48x48_hidden16/my_model_simple') # 4x4

mask = np.ones([1,8]).astype(np.float32)
imgage_dim = 48
power = 3
import cv2

#path_to_search = "c:/data/AgentData/Simple/SimpleCollector_192x192_hidden16/02.10.2024 09_45_33/"
#path_to_search = "c:/data/AgentData/Simple/SimpleCollector_128x128_hidden16/02.10.2024 09_52_45/"
#path_to_search = "c:/data/AgentData/Simple/SimpleCollector_96x96_hidden16/02.10.2024 09_59_25/"
#path_to_search = "c:/data/AgentData/Simple/SimpleCollector_64x64_hidden16/02.10.2024 10_04_25/"
#path_to_search = "c:/data/AgentData/Simple/SimpleCollector_48x48_hidden16/02.10.2024 10_09_21/"

#path_to_search = "c:/data/AgentData/Nature/SimpleCollector_192x192_hidden16/02.10.2024 08_30_56/"
path_to_search = "c:/data/AgentData/Nature/SimpleCollector_128x128_hidden16/02.10.2024 08_56_47/"
#path_to_search = "c:/data/AgentData/Nature/SimpleCollector_96x96_hidden16/02.10.2024 09_28_53/"
#path_to_search = "c:/data/AgentData/Nature/SimpleCollector_64x64_hidden16/02.10.2024 09_34_29/"
#path_to_search = "c:/data/AgentData/Nature/SimpleCollector_48x48_hidden16/02.10.2024 09_39_46/"

display_predictions = False
#set NONE if you do not want video
#saveToVideoFile = None
"""
saveToVideoFile = "c:/data/AgentData/Simple/SimpleCollector_192x192_hidden16.mp4"
saveToVideoFile = "c:/data/AgentData/Simple/SimpleCollector_128x128_hidden16.mp4"
saveToVideoFile = "c:/data/AgentData/Simple/SimpleCollector_96x96_hidden16.mp4"
saveToVideoFile = "c:/data/AgentData/Simple/SimpleCollector_64x64_hidden16.mp4"
saveToVideoFile = "c:/data/AgentData/Simple/SimpleCollector_48x48_hidden16.mp4"
"""
saveToVideoFile = "c:/data/AgentData/Nature/SimpleCollector_192x192_hidden16.mp4"
saveToVideoFile = "c:/data/AgentData/Nature/SimpleCollector_128x128_hidden16.mp4"
saveToVideoFile = "c:/data/AgentData/Nature/SimpleCollector_96x96_hidden16.mp4"
saveToVideoFile = "c:/data/AgentData/Nature/SimpleCollector_64x64_hidden16.mp4"
saveToVideoFile = "c:/data/AgentData/Nature/SimpleCollector_48x48_hidden16.mp4"
saveToVideoFile = None

FPS = 60
image_files = []

all_files_i = []
all_files_s = []
all_files_o = []
import glob, os
os.chdir(path_to_search)
for file in glob.glob("*_i.png"):
    all_files_i.append(file)
for file in glob.glob("*_s.png"):
    all_files_s.append(file)
for file in glob.glob("*_o.png"):
    all_files_o.append(file)

# initialize our gradient class activation map and build the heatmap
cam = GradCamKeras(model)

# font
font = cv2.FONT_HERSHEY_SIMPLEX
# org
org = (10, 20)
# fontScale
fontScale =0.5
# Blue color in BGR
color = (0, 0, 0)
# Line thickness of 2 px
thickness = 2


#for file_name in all_files_i:
#for file_id in range(len(all_files_i)):
for file_id in range(100, 1000):
    file_name_i = all_files_i[file_id]
    file_name_s = all_files_s[file_id]
    file_name_o = all_files_o[file_id]
    # image file
    img_file = path_to_search + file_name_i
    img = cv2.imread(img_file)
    img = cv2.resize(img, [imgage_dim,imgage_dim])
    # semantic file
    img_file = path_to_search + file_name_s
    img_s = cv2.imread(img_file)
    img_s = cv2.resize(img_s, [imgage_dim, imgage_dim])
    # overall file
    img_file = path_to_search + file_name_o
    img_o = cv2.imread(img_file)
    img_o = cv2.resize(img_o, [imgage_dim, imgage_dim])

    img_reshaped = prepareImage(img, imgage_dim)
    # load the original image from disk (in OpenCV format) and then
    # resize the image to its target dimensions
    orig = img
    resized = img
    image = img

    # use the network to make predictions on the input image and find
    # the class label index with the largest corresponding probability
    preds = model.predict([img_reshaped, mask])

    preds_argmax = returnPredictions(preds)
    print(preds_argmax)

    for output_id in range(1):

        [heatmap_INTER_NEAREST, heatmap_INTER_NEAREST_01, heatmap, heatmap_unscaled] = cam.compute_heatmap([img_reshaped, mask], output_id)

        # resize the resulting heatmap to the original input image dimensions
        # and then overlay heatmap on top of the image
        heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
        heatmap_INTER_NEAREST = cv2.resize(heatmap_INTER_NEAREST, (orig.shape[1], orig.shape[0]))
        (heatmap, output, output2) = cam.overlay_heatmap(heatmap, orig, alpha=0.5, power=power,
                                                         colormap=cv2.COLORMAP_HOT)

        (heatmap_INTER_NEAREST_color, output_INTER_NEAREST, output2_INTER_NEAREST) = cam.overlay_heatmap(heatmap_INTER_NEAREST,
                                                                            orig, alpha=0.5, power=power,
                                                                            colormap=cv2.COLORMAP_HOT)



        # display the original image and resulting heatmap and output image
        # to our screen

        output_INTER_NEAREST = np.hstack([orig, heatmap_INTER_NEAREST_color, output_INTER_NEAREST, output2_INTER_NEAREST])
        output_INTER_NEAREST = imutils.resize(output_INTER_NEAREST, width=800)
        #output_INTER_NEAREST = np.hstack([orig, heatmap, output])
        output = np.hstack([img_o, heatmap, output, output2])
        output = imutils.resize(output, width=800)

        ################################################################
        (heatmap_INTER_NEAREST_s, output_INTER_NEAREST_s, output2_INTER_NEAREST_s) = cam.overlay_heatmap(heatmap_INTER_NEAREST,
                                                                            img_s, alpha=0.5, power=power,
                                                                            colormap=cv2.COLORMAP_HOT)
        output_s = np.hstack([img_s, heatmap_INTER_NEAREST_s, output_INTER_NEAREST_s, output2_INTER_NEAREST_s])
        output_s = imutils.resize(output_s, width=800)
        ################################################################

        output = np.vstack([output_INTER_NEAREST, output_s, output])
        if display_predictions:
            output = cv2.putText(output, str(preds_argmax), org, font,
                                fontScale, color, thickness, cv2.LINE_AA)
        #cv2.imssaveToVideoFilehow("Output_INTER_NEAREST", output_INTER_NEAREST)
        cv2.imshow("GradCam " + str(output_id), output)
        if saveToVideoFile is not None:
            image_files.append(output)
    cv2.waitKey(1)

if saveToVideoFile is not None:
    import moviepy.video.io.ImageSequenceClip
    for id_img in range(len(image_files)):
        image_files[id_img] = cv2.cvtColor(image_files[id_img], cv2.COLOR_BGR2RGB)
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=FPS)
    clip.write_videofile(saveToVideoFile)
else:
    cv2.waitKey(0)