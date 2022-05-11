import tensorflow as tf
from tensorflow.keras.layers import \
    Conv2D, MaxPool2D, Dropout, Flatten, Dense

import hyperparameters as hp


class YourModel(tf.keras.Model):
    """ Your own neural network model. """

    def __init__(self):
        super(YourModel, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=hp.learning_rate)
        self.architecture = [
            Conv2D(200, 5, padding='same', activation='relu', input_shape=(hp.img_height, hp.img_width, 3)),
            MaxPool2D(),
            Dropout(.5),
            Conv2D(100, 5, padding='same', activation='relu', input_shape=(hp.img_height, hp.img_width, 3)),
            MaxPool2D(),
            Dropout(.5),
            Conv2D(50, 3, padding='same', activation='relu', input_shape=(hp.img_height, hp.img_width, 3)),
            MaxPool2D(),
            Dropout(.5),
            Flatten(),
            Dense(100),
            Dropout(.5),
            Dense(hp.num_classes, activation='softmax')
        ]


    def call(self, x):
        """ Passes input image through the network. """
        for layer in self.architecture:
            x = layer(x)
        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for the model. """
        return tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)

