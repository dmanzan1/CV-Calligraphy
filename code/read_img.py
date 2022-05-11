import tensorflow as tf
from tensorflow.python.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from skimage import io
import matplotlib.pyplot as plt




def recognize(test_img, target_height, target_width):
    """
    Takes in a sample image and cuts it into slices to predict 
    identity of each letter

    Params: 
      test_img: array representing one image of a word
      target_height: normalization height for sliced letters
      target_width: normalization width for sliced letters
    """
    tensor = tf.cast(test_img, dtype=tf.float32)
    normed = (tensor - tf.reduce_min(tensor)) / (tf.reduce_max(tensor) - tf.reduce_min(tensor))
    normed = 1 - normed
    gray_scale = tf.math.reduce_mean(normed, axis=2)
    denoise = tf.where(gray_scale < 0.2, x=0, y=1)

    word = 1 - denoise
    reduced = tf.reduce_sum(word, axis=0)

    cuts = []
    bound = 240
    width = len(reduced)

    for i in range(width):
      if i + 1 < width:
        if (reduced[i] >= bound and reduced[i+1] < bound): # start of a letter
          cuts.append((i, "end"))
        if (reduced[i] < bound and reduced[i+1] >= bound): # end of a letter
          cuts.append((i, "start"))
    
    
    slices = []
    last_cut = 0
    for cut, kind in cuts:
     if kind == "start":
        last_cut = cut
     if kind == "end":
        expanded = tf.expand_dims(word[:,last_cut:cut], -1)
        resized = tf.image.resize_with_pad(expanded,target_height,target_width)
        #resized = tf.squeeze(resized, -1)
        resized = tf.reduce_mean(resized, axis=2)
        slices.append(resized)

    return slices
        

