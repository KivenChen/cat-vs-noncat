from dnn_app_utils_v3 import *
import numpy as np

# this script reads from h5 and output them to npy files
dir = "datasets/"
train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
np.save(dir+"train_x_orig", train_x_orig)
np.save(dir+"train_y", train_y)
np.save(dir+"test_x_orig", test_x_orig)
np.save(dir+"test_y", test_y)
np.save(dir+"classes", classes)