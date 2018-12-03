'''
In this file, I will design a program to extract the files and feed a particular model
'''
import numpy as np
import tensorflow as tf
from dnn_app_utils_v3 import *
import wheels
'''loading keras 2d'''
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, \
    AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.optimizers import adam, sgd
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import keras.backend as K
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform


def train(file, x, y, epochs, batch_size=32):
    model = load_model(file)
    model = Model()
    model.fit(x, y, batch_size=batch_size, epochs=epochs)


def collect_data():

    pass
