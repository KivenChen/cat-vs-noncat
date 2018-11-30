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
from time import sleep

def make_conv_v2(size=(64, 64, 3), normalize=False):
    X_input = Input(shape=size)
    X = Conv2D(8, (16, 16), padding='same')(X_input)
    if normalize:
        X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPooling2D()(X)
    
    X = Conv2D(16, (8, 8))(X)
    if normalize:
        X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPooling2D()(X)
    
    X = Conv2D(32, (4, 4))(X)
    if normalize:
        X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPooling2D()(X)
    
    X = Conv2D(64, (2, 2))(X)
    X = Activation('relu')(X)
    X = AveragePooling2D()(X)
    X = Dense(4, activation='relu')(X)
    X = Flatten()(X)
    X = Dense(1)(X)
    
    X = Activation('sigmoid')(X)
    model = Model(X_input, X)
    model.compile(optimizer=adam(), loss='binary_crossentropy', metrics=['accuracy',])
    return model


def make_conv_model(size=(64, 64, 3), normalize=False):
    X_input = Input(shape=size)
    # first layer
    X = Conv2D(8, (8, 8), padding='same')(X_input)
    if normalize:
    	X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPooling2D()(X)  # now 16*16
    X = Dense(8, activation='relu')(X)
    X = Flatten()(X)
    X = Dense(1)(X)
    X = Activation('sigmoid')(X)
    model = Model(X_input, X)
    model.compile(optimizer=adam(), loss='binary_crossentropy', metrics=['accuracy',])
    return model
    # second conv


def make_mlp_model(lr=0.001, size=(12288,), normalize=False):
    # creates a model of fully connected layers
    # which is already compiled
    X_input = Input(shape=size)
    X = Dense(12288, activation='relu', name="fc_1")(X_input)
    if normalize:
        X = BatchNormalization()(X)
    X = Dense(20, activation='relu', name='fc_2',)(X)
    X = Dense(7, activation='relu', name="fc_3")(X)
    X = Dense(5, activation='relu', name="fc_4")(X)
    X = Dense(1, activation='relu', name="fc_final")(X)
    X = Activation('sigmoid')(X)
    model = Model(X_input, X)
    model.compile(optimizer=adam(lr=lr), loss='binary_crossentropy', metrics=['accuracy',])
    return model

max = 87

def conv_main(iterations=250, normalize=False):
    train_x_orig, train_y, test_x_orig, test_y, classes = load_data_from_npy()
    # The "-1" makes reshape flatten the remaining dimensions
    # Adapt the dims of y to fit the Keras Framework
    train_y = train_y.T
    test_y = test_y.T
    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_orig / 255.
    test_x = test_x_orig / 255.

    print("train", train_x.shape)
    print("test", test_x.shape)
    print("train_y", train_y.shape)
    model = make_conv_v2(normalize=normalize)
    for i in range(iterations):
        model.fit(train_x, train_y, verbose=1)
        if evaluate_model(model, test_x, test_y, "test set"):
            print("\n"*100, "test accrucy now:", max)
            model.save("181130, acc:"+str(max))
        print(i," th iteration")
    model.fit(train_x, train_y, batch_size=233, epochs=1)
    wheels.green("The final evaluation:")
    evaluate_model(model, test_x, test_y, "test set")


def mlp_main(iterations=2500, normalize=False):
    global max
    train_x_orig, train_y, test_x_orig, test_y, classes = load_data_from_npy()
    # The "-1" makes reshape flatten the remaining dimensions
    # There was a transposition, but it has been deleted to fit the Keras framework.
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1)  # .T
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1)
    # Adapt the dims of y to fit the Keras Framework
    train_y = train_y.T
    test_y = test_y.T
    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten / 255.
    test_x = test_x_flatten / 255.

    print("train", train_x.shape)
    print("test", test_x.shape)
    print("train_y", train_y.shape)
    model = make_mlp_model(normalize=normalize)
    for i in range(iterations):
        callback = model.fit(train_x, train_y, verbose=1)
        evaluate_model(model, train_x, train_y, "train set"),
        if evaluate_model(model, test_x, test_y, "test set"):
            model.save("conv_model")
            print("\n"*100)
            time.sleep(10)
            print("accuracy now", max)
        print(i, "th iteration")
    model.fit(train_x, train_y, batch_size=233, epochs=1)
    wheels.green("The final evaluation:")
    evaluate_model(model, test_x, test_y, "test set")
    np.savetxt('log 1107', callback)



def evaluate_model(model, x, y, name=None):
    global max
    preds = model.evaluate(x, y)
    print("the result for", name, ":")
    print("loss = ", str(preds[0]))
    print("accuracy = ", str(preds[1]))
    if preds[1]>max:
        max = preds[1]
        return True
    return False
    
if __name__ == "__main__":
    conv_main(normalize=True)
        # sgd with normalization: 76%
        # took MUCH LONGER to train than the numpy version
        # without normalization: 34%  ......
        # why this happens?
2
        # adam with normalization: 77%+
        # but the accuracy then dropped to 60% - 75%
        # the loss is not dropping that fast ... it remained very high
        # took even longer to train
        # without normalization: ?


