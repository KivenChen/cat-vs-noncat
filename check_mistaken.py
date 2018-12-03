' this module shows incorrectly-labeled images'
'may need some random brightness and random contrast and random saturnation'

from keras.models import Model, load_model
from dnn_app_utils_v3 import *
from wheels import *
from matplotlib import pyplot as plt
from keras.preprocessing import image

model_file_name = "181130-acc-0.98.h5" # this model are mostly affected by BLACK color
# model_file_name = "181130-acc-0.9600000023841858.h5" # this model are mostly affected by BRIGHT colorS
# model_file_name = "181130-acc-0.9599999904632568.h5" # same

def main():
    blue("Loading")
    _, _, test_x, test_y, _ = load_data_from_npy()
    test_y = test_y.T
    test_x = test_x / 255
    model: Model = load_model(model_file_name)
    for i in range(50):
        prob = model.predict(test_x[i].reshape((1,64,64,3)))
        result = 1 if prob>0.5 else 0
        if True:
            print(i)
            err(prob)
            plt.figure()
            plt.title(str(i))
            plt.imshow(image.array_to_img(test_x[i]))
            plt.show()
            plt.close()


if __name__ == "__main__":
    main()
