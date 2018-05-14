# -*- coding: utf-8 -*-
"""
Created on Tue May  1 09:07:29 2018

@author: aag14
"""
import keras

def intersection_over_union(output, target):
    """
    Parameters
    ----------
    output: (N, 4) ndarray of float
    target: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    target_area = (target[:, 2] - target[:, 0] + 1) * (target[:, 3] - target[:, 1] + 1)
    output_area = keras.backend.expand_dims((output[:, 2] - output[:, 0] + 1) * (output[:, 3] - output[:, 1] + 1), 1)


    intersection_c_minimum = keras.backend.minimum(keras.backend.expand_dims(output[:, 2], 1), target[:, 2])
    intersection_c_maximum = keras.backend.maximum(keras.backend.expand_dims(output[:, 0], 1), target[:, 0])

    intersection_r_minimum = keras.backend.minimum(keras.backend.expand_dims(output[:, 3], 1), target[:, 3])
    intersection_r_maximum = keras.backend.maximum(keras.backend.expand_dims(output[:, 1], 1), target[:, 1])

    intersection_c = intersection_c_minimum - intersection_c_maximum + 1
    intersection_r = intersection_r_minimum - intersection_r_maximum + 1

    intersection_c = keras.backend.maximum(intersection_c, 0)
    intersection_r = keras.backend.maximum(intersection_r, 0)
    return intersection_c
    intersection_area = intersection_c * intersection_r

    union_area = output_area + target_area - intersection_area

    union_area = keras.backend.maximum(union_area, keras.backend.epsilon())


    return intersection_area / union_area