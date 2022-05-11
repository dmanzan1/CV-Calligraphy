import tensorflow as tf
from tensorflow.python.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from skimage import io
import matplotlib.pyplot as plt




def recognize(test_img, target_height, target_width):
    tensor = tf.cast(test_img, dtype=tf.float32)
    normed = (tensor - tf.reduce_min(tensor)) / (tf.reduce_max(tensor) - tf.reduce_min(tensor))
    normed = 1 - normed
    gray_scale = tf.math.reduce_mean(normed, axis=2)
    denoise = tf.where(gray_scale < 0.2, x=0, y=1)

    omi = 1 - denoise
    reduced = tf.reduce_sum(omi, axis=0)

    cuts = []
    bound = 220
    width = len(reduced)

    for i in range(width):
        if i + 1 < width:
            if (reduced[i] >= bound and reduced[i+1] < bound): # this is end of letter
                cuts.append((i, "start"))
            if (reduced[i] < bound and reduced[i+1] >= bound):
                cuts.append((i, "end"))

    slices = []
    for cut, kind in cuts:
        if kind == "start":
            last_cut = cut
        if kind == "end":
            expanded = tf.expand_dims(omi[:,last_cut:cut], -1)
            #print(expanded.shape)
            resized = tf.image.resize_with_pad(expanded,target_height,target_width)
            #print(resized.shape)
            resized = tf.squeeze(resized, -1)
            #print(resized.shape)
            slices.append(resized)

    return slices

