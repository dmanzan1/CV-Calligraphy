import tensorflow as tf
from tensorflow.keras.layers import \
    Conv2D, MaxPool2D, Dropout, Flatten, Dense

import hyperparameters as hp


class YourModel(tf.keras.Model):
    """ Your own neural network model. """

    def __init__(self):
        super(YourModel, self).__init__()

        # TODONE: Select an optimizer for your network (see the documentation
        #       for tf.keras.optimizers)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=hp.learning_rate)

        # TODONE: Build your own convolutional neural network, using Dropout at
        #       least once. The input image will be passed through each Keras
        #       layer in self.architecture sequentially. Refer to the imports
        #       to see what Keras layers you can use to build your network.
        #       Feel free to import other layers, but the layers already
        #       imported are enough for this assignment.
        #
        #       Remember: Your network must have under 15 million parameters!
        #       You will see a model summary when you run the program that
        #       displays the total number of parameters of your network.
        #
        #       Remember: Because this is a 15-scene classification task,
        #       the output dimension of the network must be 15. That is,
        #       passing a tensor of shape [batch_size, img_size, img_size, 1]
        #       into the network will produce an output of shape
        #       [batch_size, 15].
        #
        #       Note: Keras layers such as Conv2D and Dense give you the
        #             option of defining an activation function for the layer.
        #             For example, if you wanted ReLU activation on a Conv2D
        #             layer, you'd simply pass the string 'relu' to the
        #             activation parameter when instantiating the layer.
        #             While the choice of what activation functions you use
        #             is up to you, the final layer must use the softmax
        #             activation function so that the output of your network
        #             is a probability distribution.
        #
        #       Note: Flatten is a very useful layer. You shouldn't have to
        #             explicitly reshape any tensors anywhere in your network.

        # Summary of above: ---------------------------
        # use Dropout once+
        # pick from Conv2D, MaxPool2D, Dropout, Flatten, Dense
        # < 15mil params
        # [batch_size, img_size, img_size, 1] -> [batch_size, 15]
        # final layer must use softmax
        # flatten is useful layer, don't reshape any tensors

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
            Dense(15, activation='softmax')
        ]


    def call(self, x):
        """ Passes input image through the network. """

        for layer in self.architecture:
            x = layer(x)

        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for the model. """

        # TODONE: Select a loss function for your network (see the documentation
        #       for tf.keras.losses)
        return tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)

