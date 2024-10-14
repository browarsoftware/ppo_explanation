# Implementation of Algorithm 1

"""
NW N NE
W  C E
SW S SE
"""
import math
# matrix order [w, h]
import numpy as np
import cv2

"""
w=1
0 1
w=2
0 1 2
w=3
0 1 2 3
w=4
0 1 3 4
w=5
0 2 3 5
w=6
0 2 4 6
"""
def generate_borders(w, eps=0.001):
    w_border0 = math.floor(w / 3 + eps)
    w_border1 = math.floor(2 * w / 3  + eps)
    w_borders = [0]
    for ww in [w_border0, w_border1, w]:
        if ww - w_borders[len(w_borders) - 1] > 0:
            w_borders.append(ww)
    if w == 4:
        w_borders[2] += 1
    elif w > 4:
        w_borders[1] = w_borders[3] - w_borders[2]
    return w_borders

def generate_all_borders(my_array):
    #w, h
    w = my_array.shape[0]
    h = my_array.shape[1]
    #oreder: w, h
    window_sizes = []
    window_names = []

    w_borders = generate_borders(w)
    h_borders = generate_borders(h)

    for w_id in range(len(w_borders) - 1):
        w_start = w_borders[w_id]
        w_stop = w_borders[w_id + 1]
        for h_id in range(len(h_borders) - 1):
            h_start = h_borders[h_id]
            h_stop = h_borders[h_id + 1]
            window_sizes.append([[w_start, w_stop], [h_start, h_stop]])

    if len(w_borders) == 1 + 1 and len(h_borders)  == 1 + 1:
        window_names = ["C"]
    elif len(w_borders) == 1 + 1 and len(h_borders)  == 2 + 1:
        window_names = ["N", "S"]
    elif len(w_borders) == 1 + 1 and len(h_borders)  == 3 + 1:
        window_names = ["N", "C", "S"]
    elif len(w_borders) == 2 + 1 and len(h_borders)  == 1 + 1:
        window_names = ["W", "E"]
    elif len(w_borders) == 2 + 1 and len(h_borders)  == 2 + 1:
        window_names = ["NW", "NE", "SW", "SE"]
    elif len(w_borders) == 2 + 1 and len(h_borders)  == 3 + 1:
        window_names = ["NW", "W", "SW", "NE", "E", "SE"]
    elif len(w_borders) == 3 + 1 and len(h_borders)  == 1 + 1:
        window_names = ["W", "C", "E"]
    elif len(w_borders) == 3 + 1 and len(h_borders)  == 2 + 1:
        window_names = ["NW", "N", "NE",
                            "SW", "S", "SE"]
    elif len(w_borders) == 3 + 1 and len(h_borders)  == 3 + 1:
        window_names = ["NW", "N", "NE",
                            "W", "C", "E",
                            "SW", "S", "SE"]
    return (window_sizes, window_names)
