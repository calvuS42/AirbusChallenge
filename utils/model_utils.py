"""Define the model architecture, metrics and inference functions.

    This script requires 'os', 'numpy', 'tensorflow', 'matplotlib'
    to be installed within the Python environment
    you are running this script in.

    This file should be imported as a module and contains the following
    class and functions:

    * UNET - basic model architecture that is used for training.
    * dice_coef - Dice coef function.
    * dice_coef_loss - Dice loss function that is equal to 1 - Dice coef.
    * IoU - IoU metrics function.
    * plot_history - Plot the train and val loss over training epochs.
    * predict - Predict mask for image by model.
"""

import os
from tensorflow.keras import layers
from tensorflow import keras as K
import numpy as np
from matplotlib import pyplot as plt


class UNET:
    '''
    A class for defining the model architecture for training.

    ...

    Attributes
    ----------
    model : tf.keras.Model
        model object that is created with input parameters

    Methods
    -------
    _build_model(num_classes, input_shape)
        creating UNET-like model
    '''

    def __init__(self, num_classes: int, input_shape=(768, 768, 3)):
        '''Initialize class instance.

        Parameters
        ----------
        num_classes : int
            number of classes for model to predict.
        input_shape : tuple, optional
            (height,width,depth) of the input image for model.
        '''
        self._model = self._build_model(num_classes, input_shape)

    def _build_model(self, num_classes, input_shape):
        '''Create model with input parameters.

        Parameters
        ----------
        num_classes : int
            number of classes for model to predict.
        input_shape : tuple, optional
            (height,width,depth) of the input image for model.

        Returns
        -------
        tf.keras.Model
            model with Unet-like architecture
        '''
        inputs = layers.Input(shape=input_shape)

        # encoder: contracting path - downsample
        # 1 - downsample
        f1, p1 = _downsample_block(inputs, 16)
        # 2 - downsample
        f2, p2 = _downsample_block(p1, 32)
        # 3 - downsample
        f3, p3 = _downsample_block(p2, 64)
        # 4 - bottleneck
        bottleneck = _double_conv_block(p3, 128)

        # decoder: expanding path - upsample
        # 5 - upsample
        u5 = _upsample_block(bottleneck, f3, 64)
        # 6 - upsample
        u6 = _upsample_block(u5, f2, 32)
        # 7 - upsample
        u7 = _upsample_block(u6, f1, 16)

        # outputs
        outputs = layers.Conv2D(num_classes, 1, padding="same",
                                activation="softmax")(u7)

        # unet model with Keras Functional API
        unet_model = K.Model(inputs, outputs, name="U-Net")

        return unet_model

    @property
    def model(self):
        '''Returns _model field.'''
        return self._model


def _double_conv_block(x, n_filters):
    '''Define the double convolution block.'''
    # Conv2D then ReLU activation
    x = layers.Conv2D(
        n_filters,
        3,
        padding="same",
        activation='relu',
        kernel_initializer="he_normal")(x)

    # Conv2D then ReLU activation
    x = layers.Conv2D(
        n_filters,
        3,
        padding="same",
        activation="relu",
        kernel_initializer="he_normal")(x)

    return x


def _downsample_block(x, n_filters):
    '''Define the downsample block.'''
    f = _double_conv_block(x, n_filters)
    p = layers.MaxPool2D(2)(f)
    p = layers.Dropout(0.3)(p)
    return f, p


def _upsample_block(x, conv_features, n_filters):
    '''Define the upwnsample block.'''
    # upsample
    x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
    # concatenate
    x = layers.concatenate([x, conv_features])
    # dropout
    x = layers.Dropout(0.3)(x)
    # Conv2D twice with ReLU activation
    x = _double_conv_block(x, n_filters)
    return x


def dice_coef(y_true, y_pred, smooth=1.):
    '''Dice coef function.

    Parameters
    ----------
    y_true : Tensor
        ground-truth mask.
    y_pred : Tensor
        predicted mask.
    smooth: float
        coeficient for smoothing Dice coef.

    Returns
    -------
    float
        dice coef for two arrays

    Reference
    ---------
    https://www.kaggle.com/code/yerramvarun/understanding-dice-coefficient
    https://www.kaggle.com/code/iafoss/unet34-dice-0-87
    https://stackoverflow.com/questions/72195156/correct-implementation-of-dice-loss-in-tensorflow-keras
    '''
    y_true_f = K.backend.flatten(y_true)
    y_pred_f = K.backend.flatten(y_pred)
    intersection = K.backend.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / \
        (K.backend.sum(y_true_f) + K.backend.sum(y_pred_f) + smooth)
    return dice


def dice_coef_loss(y_true, y_pred):
    '''Dice loss function that is equal to 1 - Dice coef.

    Parameters
    ----------
    y_true : Tensor
        ground-truth mask.
    y_pred : Tensor
        predicted mask.

    Returns
    -------
    float
        dice loss for two arrays
    '''
    return 1-dice_coef(y_true, y_pred)


def IoU(y_pred, y_true, smooth=1.):
    '''Inetsection over Union metrics function.

    Parameters
    ----------
    y_true : Tensor
        ground-truth mask.
    y_pred : Tensor
        predicted mask.
    smooth: float
        coeficient for smoothing IoU.

    Returns
    -------
    float
        IoU for two arrays
    '''
    y_true_f = K.backend.flatten(y_true)
    y_pred_f = K.backend.flatten(y_pred)
    intersection = K.backend.sum(y_true_f * y_pred_f)
    return intersection / \
        (K.backend.sum(y_true_f+y_pred_f) - intersection + smooth)


def plot_history(hist, path):
    '''Plot the train and val loss over training epochs.

    Parameters
    ----------
    hist
        output of model.fit.
    path
        path to save plot.
    '''
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']

    plt.figure()
    plt.plot(hist.epoch, loss, 'g', label='Training loss')
    plt.plot(hist.epoch, val_loss, 'r', label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.savefig(os.path.join(path, 'model_history.png'))
    plt.show()


def predict(image, model):
    '''Predict mask for image by model.

    Parameters
    ----------
    image
        image in format that Keras could process.
    model : tf.Keras.Model
        model that should be inferenced.

    Returns
    -------
    numpy array
        predicted mask, 1 - mask, 0 - background
    '''
    image = np.expand_dims(image, axis=0)
    pred_mask = model.predict(image)[0].argmax(axis=-1)
    return pred_mask
