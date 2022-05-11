"""
Homework 5 - CNNs
CS1430 - Computer Vision
Brown University
"""
import matplotlib.pyplot as plt
import os
import random
import numpy as np
from PIL import Image
import tensorflow as tf
import hyperparameters as hp

class Datasets():
    """ Class for containing the training and test sets as well as
    other useful data-related information. Contains the functions
    for preprocessing.
    """

    def __init__(self, data_path):

        self.data_path = data_path
        
        # Dictionaries for (label index) <--> (class name)
        self.idx_to_class = {}
        self.class_to_idx = {}

        # For storing list of classes
        self.classes = [""] * hp.num_classes
        
        # Setup data generators
        self.train_data = self.get_data(
            os.path.join(self.data_path, "train/"),  True, True)
        self.test_data = self.get_data(
            os.path.join(self.data_path, "test/"),  False, False)

    def preprocess_fn(self, img):
        """ Preprocess function for ImageDataGenerator. """
        img = tf.cast(img, dtype=tf.float32)
        normed = ((img - tf.reduce_min(img)) / 
                  (tf.reduce_max(img) - tf.reduce_min(img)))
        inverted = 1 - normed
        resized = tf.image.resize_with_pad(inverted, hp.img_height, hp.img_width)
        denoise = tf.where(resized < 0.2, x=0, y=1)
        img = 1 - denoise 

        # plt.imshow(img, cmap='gray')
        return img

    def get_data(self, path, shuffle, augment):
        """ Returns an image data generator which can be iterated
        through for images and corresponding class labels.

        Arguments:
            path - Filepath of the data being imported, such as
                   "../data/train" or "../data/test"
            shuffle - Boolean value indicating whether the data should
                      be randomly shuffled.
            augment - Boolean value indicating whether the data should
                      be augmented or not.

        Returns:
            An iterable image-batch generator
        """

        data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
                preprocessing_function=self.preprocess_fn,
                width_shift_range=5
                )

        classes_for_flow = None

        # Make sure all data generators are aligned in label indices
        if bool(self.idx_to_class):
            classes_for_flow = self.classes

        # Form image data generator from directory structure
        data_gen = data_gen.flow_from_directory(
            path,
            target_size=(hp.img_height, hp.img_width),
            class_mode='sparse',
            batch_size=hp.batch_size,
            shuffle=shuffle,
            classes=classes_for_flow)

        # Setup the dictionaries if not already done
        if not bool(self.idx_to_class):
            unordered_classes = []
            for dir_name in os.listdir(path):
                if os.path.isdir(os.path.join(path, dir_name)):
                    unordered_classes.append(dir_name)

            for img_class in unordered_classes:
                self.idx_to_class[data_gen.class_indices[img_class]] = img_class
                self.class_to_idx[img_class] = int(data_gen.class_indices[img_class])
                self.classes[int(data_gen.class_indices[img_class])] = img_class

        return data_gen