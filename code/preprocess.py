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

    def __init__(self, data_path, task):

        self.data_path = data_path
        self.task = task

        # Dictionaries for (label index) <--> (class name)
        self.idx_to_class = {}
        self.class_to_idx = {}

        # For storing list of classes
        self.classes = [""] * hp.num_classes

        # Mean and std for standardization
        self.mean = np.zeros((hp.img_height,hp.img_width,3))
        self.std = np.ones((hp.img_height,hp.img_width,3))
        self.calc_mean_and_std()

        # Setup data generators
        self.train_data = self.get_data(
            os.path.join(self.data_path, "train/"), task == '3', True, True)
        self.test_data = self.get_data(
            os.path.join(self.data_path, "test/"), task == '3', False, False)

    def calc_mean_and_std(self):
        """ Calculate mean and standard deviation of a sample of the
        training dataset for standardization.

        Args: none
        Return: none
        """

        # Get list of all images in training directory
        file_list = []
        for root, _, files in os.walk(os.path.join(self.data_path, "train/")):
            for name in files:
                if name.endswith(".jpg"):
                    file_list.append(os.path.join(root, name))

        # Shuffle filepaths
        random.shuffle(file_list)

        # Take sample of file paths
        file_list = file_list[:hp.preprocess_sample_size]

        # Allocate space in memory for images
        data_sample = np.zeros(
            (hp.preprocess_sample_size, hp.img_height,hp.img_width, 3))

        # Import images
        for i, file_path in enumerate(file_list):
            img = Image.open(file_path)
            img = img.resize((hp.img_height,hp.img_width))
            img = np.array(img, dtype=np.float32)
            img /= 255.

            # # Grayscale -> RGB
            # if len(img.shape) == 2:
            #     img = np.stack([img, img, img], axis=-1)

            data_sample[i] = img

        # TASK 1
        # TODONE: Calculate the pixel-wise mean and standard deviation
        #       of the images in data_sample and store them in
        #       self.mean and self.std respectively.
        # (we want to find the "mean image")

        self.mean = np.mean(data_sample, axis=0)
        self.std = np.std(data_sample, axis=0)

    def standardize(self, img):
        """ Function for applying standardization to an input image.

        Arguments: img - np array of shape (image size, image size, 3)

        Returns: img - np array of same initial size
        """
        img = (img - self.mean) / self.std
        return img

    def preprocess_fn(self, img):
        """ Preprocess function for ImageDataGenerator. """
        img = img / 255.
        img = self.standardize(img)

        tensor = tf.cast(img, dtype=tf.float32)
        inverted = 1 - tensor
        resized = tf.image.resize_with_pad(inverted, hp.img_height, hp.img_width)
        #**changedfrom 200 242 into 224 224
        denoise = tf.where(resized < 0.2, x=0, y=1)
        img = 1 - denoise 
        
        plt.imshow(img, cmap='gray')


        return img

    def get_data(self, path, is_vgg, shuffle, augment):
        """ Returns an image data generator which can be iterated
        through for images and corresponding class labels.

        Arguments:
            path - Filepath of the data being imported, such as
                   "../data/train" or "../data/test"
            is_vgg - Boolean value indicating whether VGG preprocessing
                     should be applied to the images.
            shuffle - Boolean value indicating whether the data should
                      be randomly shuffled.
            augment - Boolean value indicating whether the data should
                      be augmented or not.

        Returns:
            An iterable image-batch generator
        """

        data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=self.preprocess_fn)

        # VGG must take images of size 224x224
        #img_size = hp.img_size

        classes_for_flow = None

        # Make sure all data generators are aligned in label indices
        if bool(self.idx_to_class):
            classes_for_flow = self.classes

        # Form image data generator from directory structure
        data_gen = data_gen.flow_from_directory(
            path,
            #** possible error swap height and width
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