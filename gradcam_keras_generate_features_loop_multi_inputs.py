# Generate data to evaluate approach
import os
from GradCamKeras import GradCamKeras
import imutils
import tensorflow as tf
import numpy as np
from BordersGenerator import generate_all_borders

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

from numba import  jit
@jit(nopython=True)
def simulator(pair00, pair01, pair10, pair11, img, pixel_color, scale_by_heatmap,
              heatmap_INTER_NEAREST_01, power, threshold, results,
              window_sizes_id, colors_map_len, color_map_id):
    for w_id in range(pair00, pair01):
        for h_id in range(pair10, pair11):
            if img[w_id, h_id, 2] == pixel_color[0] \
                    and img[w_id, h_id, 1] == pixel_color[1] \
                    and img[w_id, h_id, 0] == pixel_color[2]:
                if scale_by_heatmap:
                    heatmap_value = math.pow(heatmap_INTER_NEAREST_01[w_id, h_id], power)

                    if heatmap_value >= threshold:
                        heatmap_value = heatmap_value
                    else:
                        heatmap_value = 0
                    results[window_sizes_id * colors_map_len + color_map_id] += 1 * heatmap_value
                else:
                    results[window_sizes_id * colors_map_len + color_map_id] += 1
    return results

def calculate_statistics(img, heatmap_INTER_NEAREST_01, window_sizes_resized, window_names, colors_map,
                         scale_by_heatmap=True, scale_by_window_szie=True, power=3, threshold=0.5):
    results = np.zeros(len(window_names) * len(colors_map))
    window_sizes_id = 0
    colors_map_len = len(colors_map)

    for pair in window_sizes:
        color_map_id = 0
        for cm in colors_map:
            pixel_color = cm[0]
            """
            pixel_color = cm[0]
            for w_id in range(pair[0][0], pair[0][1]):
                for h_id in range(pair[1][0], pair[1][1]):
                    if img[w_id,h_id,2] == pixel_color[0] \
                            and img[w_id, h_id, 1] == pixel_color[1] \
                            and img[w_id, h_id, 0] == pixel_color[2]:
                        if scale_by_heatmap:
                            heatmap_value = math.pow(heatmap_INTER_NEAREST_01[w_id,h_id], power)

                            if heatmap_value >= threshold:
                                heatmap_value = heatmap_value
                            else:
                                heatmap_value = 0
                            results[window_sizes_id * colors_map_len + color_map_id] += 1 * heatmap_value
                        else:
                            results[window_sizes_id * colors_map_len + color_map_id] += 1
            ###############################################
            """
            results = simulator(pair[0][0], pair[0][1], pair[1][0], pair[1][1], img, pixel_color, scale_by_heatmap,
                  heatmap_INTER_NEAREST_01, power, threshold, results,
                  window_sizes_id, colors_map_len, color_map_id)
            color_map_id += 1
        if scale_by_window_szie:
            #win_size =  (pair[0][1] - pair[0][0]) * (pair[1][1] - pair[1][0])
            denominator = 0
            for a in range(window_sizes_id * colors_map_len, window_sizes_id * colors_map_len + colors_map_len):
                denominator += results[a]
            if denominator > 0:
                for a in range(window_sizes_id * colors_map_len, window_sizes_id * colors_map_len + colors_map_len):
                    results[a] /= denominator
        window_sizes_id += 1
    xxss = np.sum(results)
    return results

power = 3
threshold = 0.5

power  = 1
threshold = 0

scale_by_heatmap = True
"""
all_models = ['C:/projects/python/pythonProjectOnnxToTensorflow/brains/nature192x192_hidden16/my_model_nature',
              'C:/projects/python/pythonProjectOnnxToTensorflow/brains/nature128x128_hidden16/my_model_nature',
                'C:/projects/python/pythonProjectOnnxToTensorflow/brains/nature96x96_hidden16/my_model_nature',
              'C:/projects/python/pythonProjectOnnxToTensorflow/brains/nature64x64_hidden16/my_model_nature',
              'C:/projects/python/pythonProjectOnnxToTensorflow/brains/nature56x56_hidden16/my_model_nature',
                'C:/projects/python/pythonProjectOnnxToTensorflow/brains/nature48x48_hidden16/my_model_nature']

all_paths_to_search = ["c:/data/AgentData/Nature/SimpleCollector_192x192_hidden16/02.10.2024 08_30_56/" ,
                        "c:/data/AgentData/Nature/SimpleCollector_128x128_hidden16/02.10.2024 08_56_47/",
                        "c:/data/AgentData/Nature/SimpleCollector_96x96_hidden16/02.10.2024 09_28_53/" ,
                        "c:/data/AgentData/Nature/SimpleCollector_64x64_hidden16/02.10.2024 09_34_29/",
                        "c:/data/AgentData/Nature/SimpleCollector_56x56_hidden16/04.10.2024 08_35_07/",
                       "c:/data/AgentData/Nature/SimpleCollector_48x48_hidden16/02.10.2024 09_39_46/"]


all_image_dims = [192, 128, 96, 64, 56, 48]
all_windows_sizes = [20, 12, 8, 4, 3, 2]
"""


all_models = ['C:/projects/python/pythonProjectOnnxToTensorflow/brains/simple192x192_hidden16/my_model_simple',
              'C:/projects/python/pythonProjectOnnxToTensorflow/brains/simple128x128_hidden16/my_model_simple',
                'C:/projects/python/pythonProjectOnnxToTensorflow/brains/simple96x96_hidden16/my_model_simple',
              'C:/projects/python/pythonProjectOnnxToTensorflow/brains/simple64x64_hidden16/my_model_simple',
                'C:/projects/python/pythonProjectOnnxToTensorflow/brains/simple56x56_hidden16/my_model_simple',
                'C:/projects/python/pythonProjectOnnxToTensorflow/brains/simple48x48_hidden16/my_model_simple']

all_paths_to_search = ["c:/data/AgentData/Simple/SimpleCollector_192x192_hidden16/02.10.2024 09_45_33/" ,
                        "c:/data/AgentData/Simple/SimpleCollector_128x128_hidden16/02.10.2024 09_52_45/",
                        "c:/data/AgentData/Simple/SimpleCollector_96x96_hidden16/02.10.2024 09_59_25/" ,
                        "c:/data/AgentData/Simple/SimpleCollector_64x64_hidden16/02.10.2024 10_04_25/",
                        "c:/data/AgentData/Simple/SimpleCollector_56x56_hidden16/04.10.2024 08_32_26/",
                       "c:/data/AgentData/Simple/SimpleCollector_48x48_hidden16/02.10.2024 10_09_21/"]
"""
all_save_file_paths = ["C:/data/AgentData/Simple/192x192_hidden16_power=" + str(power) + "_threshold=" + str(threshold) + "_1000.txt",
                       "C:/data/AgentData/Simple/128x128_hidden16_power=" + str(power) + "_threshold=" + str(threshold) + "_1000.txt",
                       "C:/data/AgentData/Simple/96x96_hidden16_power=" + str(power) + "_threshold=" + str(threshold) + "_1000.txt",
                       "C:/data/AgentData/Simple/64x64_hidden16_power=" + str(power) + "_threshold=" + str(threshold) + "_1000.txt",
                        "C:/data/AgentData/Simple/56x56_hidden16_power=" + str(power) + "_threshold=" + str(threshold) + "_1000.txt",
                       "C:/data/AgentData/Simple/48x48_hidden16_power=" + str(power) + "_threshold=" + str(threshold) + "_1000.txt"]
"""
all_image_dims = [192, 128, 96, 64, 56, 48]
all_windows_sizes = [22, 14, 10, 6, 5, 4]

scale_by_heatmap = True
scale_by_heatmap = True
for power in [1,2,3,5]:
    for threshold in [0,0.1,0.2,0.3,0.4,0.5]:
#for power in [1]:
#    for threshold in [0]:
        #for power in [1]:
        #    for threshold in [0]:

        all_save_file_paths = ["C:/data/AgentData/Simple/192x192_hidden16_power=" + str(power) + "_threshold=" + str(
            threshold) + "_1000.txt",
       "C:/data/AgentData/Simple/128x128_hidden16_power=" + str(power) + "_threshold=" + str(
           threshold) + "_1000.txt",
       "C:/data/AgentData/Simple/96x96_hidden16_power=" + str(power) + "_threshold=" + str(
           threshold) + "_1000.txt",
       "C:/data/AgentData/Simple/64x64_hidden16_power=" + str(power) + "_threshold=" + str(
           threshold) + "_1000.txt",
       "C:/data/AgentData/Simple/56x56_hidden16_power=" + str(power) + "_threshold=" + str(
           threshold) + "_1000.txt",
       "C:/data/AgentData/Simple/48x48_hidden16_power=" + str(power) + "_threshold=" + str(
           threshold) + "_1000.txt"]

        """
        all_save_file_paths = ["C:/data/AgentData/Nature/192x192_hidden16_power=" + str(power) + "_threshold=" + str(
        threshold) + "_1000.txt",
                           "C:/data/AgentData/Nature/128x128_hidden16_power=" + str(power) + "_threshold=" + str(
                               threshold) + "_1000.txt",
                           "C:/data/AgentData/Nature/96x96_hidden16_power=" + str(power) + "_threshold=" + str(
                               threshold) + "_1000.txt",
                           "C:/data/AgentData/Nature/64x64_hidden16_power=" + str(power) + "_threshold=" + str(
                               threshold) + "_1000.txt",
                            "C:/data/AgentData/Nature/56x56_hidden16_power=" + str(power) + "_threshold=" + str(
                               threshold) + "_1000.txt",
                           "C:/data/AgentData/Nature/48x48_hidden16_power=" + str(power) + "_threshold=" + str(
                               threshold) + "_1000.txt"]
        """
        #for my_id in([4]):
        for my_id in(range(0,len(all_models))):
            model_path = all_models[my_id]
            imgage_dim = all_image_dims[my_id]
            windows_size = all_windows_sizes[my_id]
            save_file_path = all_save_file_paths[my_id]
            path_to_search = all_paths_to_search[my_id]

            model = tf.keras.models.load_model(model_path)

            ############################################################################
            mask = np.ones([1,8]).astype(np.float32)
            (window_sizes, window_names) = generate_all_borders(my_array=np.zeros([windows_size,windows_size]))
            window_sizes_resized = copy.deepcopy(window_sizes)
            for pair in window_sizes:
                pair[0][0] = math.floor(pair[0][0] * imgage_dim / windows_size)
                pair[0][1] = math.floor(pair[0][1] * imgage_dim / windows_size)
                pair[1][0] = math.floor(pair[1][0] * imgage_dim / windows_size)
                pair[1][1] = math.floor(pair[1][1] * imgage_dim / windows_size)

            colors_map = [[(0,0,0), "Sky"], [(0,96,107), "Ground"], [(107,0,4), "Platform"], [(18,0,107), "Collectible"]]

            import cv2
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



            with open(save_file_path + "fb.txt", "a") as myfile:
                myfile.write("FileId,")
                for wn in window_names:
                    for cm in colors_map:
                        myfile.write(wn + "_" + cm[1] + ",")
                myfile.write("Forward,")
                myfile.write("Turn,")
                myfile.write("Jump\n")

            with open(save_file_path + "lr.txt", "a") as myfile:
                myfile.write("FileId,")
                for wn in window_names:
                    for cm in colors_map:
                        myfile.write(wn + "_" + cm[1] + ",")
                myfile.write("Forward,")
                myfile.write("Turn,")
                myfile.write("Jump\n")

            with open(save_file_path + "gj.txt", "a") as myfile:
                myfile.write("FileId,")
                for wn in window_names:
                    for cm in colors_map:
                        myfile.write(wn + "_" + cm[1] + ",")
                myfile.write("Forward,")
                myfile.write("Turn,")
                myfile.write("Jump\n")

            #for file_name in all_files_i:
            #for file_id in range(len(all_files_i)):
            for file_id in range(100,1100):
                print("File: " + str(file_id) + " of " + str(len(all_files_i)))
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
                img_s = cv2.resize(img_s, [imgage_dim, imgage_dim], interpolation=cv2.INTER_NEAREST)
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
                #print(preds_argmax)

                #for output_id in range(3):
                ###########################################################
                [heatmap_INTER_NEAREST, heatmap_INTER_NEAREST_01, heatmap, heatmap_unscaled] = cam.compute_heatmap([img_reshaped, mask], 0)

                results = calculate_statistics(img_s, heatmap_INTER_NEAREST_01, window_sizes_resized, window_names, colors_map,
                                               power=power, threshold=threshold, scale_by_heatmap=scale_by_heatmap)

                with open(save_file_path + "fb.txt", "a") as myfile:
                    myfile.write(str(file_id) + ",")
                    for a in range(results.shape[0]):
                        myfile.write(str(results[a]) + ",")


                    myfile.write(str(preds_argmax[0]) + ",")
                    myfile.write(str(preds_argmax[1]) + ",")
                    myfile.write(str(preds_argmax[2]) + "\n")

                ###########################################################
                [heatmap_INTER_NEAREST, heatmap_INTER_NEAREST_01, heatmap, heatmap_unscaled] = cam.compute_heatmap(
                    [img_reshaped, mask], 1)

                results = calculate_statistics(img_s, heatmap_INTER_NEAREST_01, window_sizes_resized, window_names, colors_map,
                                               power=power, threshold=threshold, scale_by_heatmap=scale_by_heatmap)

                with open(save_file_path + "lr.txt", "a") as myfile:
                    myfile.write(str(file_id) + ",")
                    for a in range(results.shape[0]):
                        myfile.write(str(results[a]) + ",")


                    myfile.write(str(preds_argmax[0]) + ",")
                    myfile.write(str(preds_argmax[1]) + ",")
                    myfile.write(str(preds_argmax[2]) + "\n")

                ###########################################################
                [heatmap_INTER_NEAREST, heatmap_INTER_NEAREST_01, heatmap, heatmap_unscaled] = cam.compute_heatmap(
                    [img_reshaped, mask], 2)

                results = calculate_statistics(img_s, heatmap_INTER_NEAREST_01, window_sizes_resized, window_names, colors_map,
                                               power=power, threshold=threshold, scale_by_heatmap=scale_by_heatmap)

                with open(save_file_path + "gj.txt", "a") as myfile:
                    myfile.write(str(file_id) + ",")
                    for a in range(results.shape[0]):
                        myfile.write(str(results[a]) + ",")


                    myfile.write(str(preds_argmax[0]) + ",")
                    myfile.write(str(preds_argmax[1]) + ",")
                    myfile.write(str(preds_argmax[2]) + "\n")
