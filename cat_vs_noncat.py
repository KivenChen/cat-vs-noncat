'''NOTE: finally comes iOS Pythonista support for TESTING ONLY. iOS Training module still under construction (0727-18, Fri)
'''

# coding: utf-8

# # Deep Neural Network for Image Classification: Application

# Let's get started!

# ## 1 - Packages

toimport = '''
Let's first import all the packages that you will need during this assignment. 
# - [numpy](www.numpy.org) is the fundamental package for scientific computing with Python.
# - [matplotlib](http://matplotlib.org) is a library to plot graphs in Python.
# - [h5py](http://www.h5py.org) is a common package to interact with a datasets that is stored on an H5 file.
# - [PIL](http://www.pythonware.com/products/pil/) and [scipy](https://www.scipy.org/) are used here to test our model with your own picture at the end.
# - dnn_app_utils provides the functions implemented in the "Building your Deep Neural Network: Step by Step" assignment to this notebook.
# - np.random.seed(1) is used to keep all the random function calls consistent. It will help us grade your work.
'''

global_lambd = 0

import time
import numpy as np
import matplotlib.pyplot as plt
'''try:
	#import scipy
	from scipy import ndimage
except ImportError:
	print('Main: detected no support for ndimage, switched to Pillow')
'''
from dnn_app_utils_v3 import *
import wheels
from wheels import *



# The following code will show you an image in the datasets. Feel free to change the index and re-run the cell multiple times to see other images.

# In[3]:

# Example of a picture
# plt.imshow(train_x_orig[index])
# print ("y = " + str(train_y[0,index]) + ". It's a " + classes[train_y[0,index]].decode("utf-8") +  " picture.")

# In[4]:

# Explore your datasets

# **Question**:  Use the helper functions you have implemented in the previous assignment to build a 2-layer neural network with the following structure: *LINEAR -> RELU -> LINEAR -> SIGMOID*. The functions you may need and their inputs are:
# ```python
# def initialize_parameters(n_x, n_h, n_y):
#     ...
#     return parameters 
# def linear_activation_forward(A_prev, W, b, activation):
#     ...
#     return A, cache
# def compute_cost(AL, Y):
#     ...
#     return cost
# def linear_activation_backward(dA, cache, activation):
#     ...
#     return dA_prev, dW, db
# def update_parameters(parameters, grads, learning_rate):
#     ...
#     return parameters
# ```

# In[6]:

### CONSTANTS DEFINING THE MODEL ####


# In[7]:

# GRADED FUNCTION: two_layer_model

def two_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    """
    Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.
    
    Arguments:
    X -- input data, of shape (n_x, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- dimensions of the layers (n_x, n_h, n_y)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- If set to True, this will print the cost every 100 iterations 
    
    Returns:
    parameters -- a dictionary containing W1, W2, b1, and b2
    """
    
    np.random.seed(1)
    grads = {}
    costs = []                              # to keep track of the cost
    m = X.shape[1]                           # number of examples
    (n_x, n_h, n_y) = layers_dims
    
    # Initialize parameters dictionary, by calling one of the functions you'd previously implemented
    ### START CODE HERE ### (≈ 1 line of code)
    parameters = initialize_parameters(n_x, n_h, n_y)
    ### END CODE HERE ###
    
    # Get W1, b1, W2 and b2 from the dictionary parameters.
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    # Loop (gradient descent)

    for i in range(0, num_iterations):

        # Forward propagation: LINEAR -> RELU -> LINEAR -> SIGMOID. Inputs: "X, W1, b1, W2, b2". Output: "A1, cache1, A2, cache2".
        ### START CODE HERE ### (≈ 2 lines of code)
        A1, cache1 = linear_activation_forward(X, W1, b1, activation="relu")
        A2, cache2 = linear_activation_forward(A1, W2, b2, activation="sigmoid")
        ### END CODE HERE ###
        
        # Compute cost
        ### START CODE HERE ### (≈ 1 line of code)
        cost = compute_cost(A2, Y, parameters, lambd = global_lambd)
        ### END CODE HERE ###
        
        # Initializing backward propagation
        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
        
        # Backward propagation. Inputs: "dA2, cache2, cache1". Outputs: "dA1, dW2, db2; also dA0 (not used), dW1, db1".
        ### START CODE HERE ### (≈ 2 lines of code)
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, activation="sigmoid", lambd = global_lambd)
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, activation="relu", lambd = global_lambd)
        ### END CODE HERE ###
        
        # Set grads['dWl'] to dW1, grads['db1'] to db1, grads['dW2'] to dW2, grads['db2'] to db2
        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2
        
        # Update parameters.
        ### START CODE HERE ### (approx. 1 line of code)
        parameters = update_parameters(parameters, grads, learning_rate)
        ### END CODE HERE ###

        # Retrieve W1, b1, W2, b2 from parameters
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        
        # Print the cost every 100 training example
        if print_cost and i % 10 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if print_cost and i % 10 == 0:
            costs.append(cost)
       
    # plot the cost

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    # plt.show()
    
    return parameters


# Run the cell below to train your parameters. See if your model runs. The cost should be decreasing. It may take up to 5 minutes to run 2500 iterations. Check if the "Cost after iteration 0" matches the expected output below, if not click on the square (⬛) on the upper bar of the notebook to stop the cell and try to find your error.

# In[8]:
'''
blue("testing two layered model")
parameters = two_layer_model(train_x, train_y, layers_dims = (n_x, n_h, n_y), num_iterations = 2500, print_cost=True)


# Good thing you built a vectorized implementation! Otherwise it might have taken 10 times longer to train this.
# 
# Now, you can use the trained parameters to classify images from the datasets. To see your predictions on the training and test sets, run the cell below.

# In[9]:

predictions_train = predict(train_x, train_y, parameters)

# In[10]:

predictions_test = predict(test_x, test_y, parameters)

green("completed\n")
'''
# **Note**: You may notice that running the model on fewer iterations (say 1500) gives better accuracy on the test set. This is called "early stopping" and we will talk about it in the next course. Early stopping is a way to prevent overfitting. 
# 
# Congratulations! It seems that your 2-layer neural network has better performance (72%) than the logistic regression implementation (70%, assignment week 2). Let's see if you can do even better with an $L$-layer model.

# ## 5 - L-layer Neural Network
# 
# **Question**: Use the helper functions you have implemented previously to build an $L$-layer neural network with the following structure: *[LINEAR -> RELU]$\times$(L-1) -> LINEAR -> SIGMOID*. The functions you may need and their inputs are:
# ```python
# def initialize_parameters_deep(layers_dims):
#     ...
#     return parameters 
# def L_model_forward(X, parameters):
#     ...
#     return AL, caches
# def compute_cost(AL, Y):
#     ...
#     return cost
# def L_model_backward(AL, Y, caches):
#     ...
#     return grads
# def update_parameters(parameters, grads, learning_rate):
#     ...
#     return parameters
# ```

# In[11]:

### CONSTANTS ###

# In[12]:

# GRADED FUNCTION: L_layer_model

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 20000, print_cost=False):#lr was 0.009
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    costs = []                         # keep track of cost
    
    # Parameters initialization. (≈ 1 line of code)
    ### START CODE HERE ###
    parameters = initialize_parameters_deep(layers_dims)
    ### END CODE HERE ###
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        ### START CODE HERE ### (≈ 1 line of code)
        AL, caches = L_model_forward(X, parameters)
        ### END CODE HERE ###
        
        # Compute cost.
        ### START CODE HERE ### (≈ 1 line of code)
        cost = compute_cost(AL, Y, parameters, lambd = global_lambd)
        ### END CODE HERE ###
    
        # Backward propagation.
        ### START CODE HERE ### (≈ 1 line of code)
        grads = L_model_backward(AL, Y, caches, lambd = global_lambd)
        ### END CODE HERE ###
 
        # Update parameters.
        ### START CODE HERE ### (≈ 1 line of code)
        parameters = update_parameters(parameters, grads, learning_rate)
        ### END CODE HERE ###
                
        # Print the cost every 100 training example
        if print_cost and i % 10 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 10 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    # plt.show()
    
    return parameters


# You will now train the model as a 4-layer neural network. 
# 
# Run the cell below to train your model. The cost should decrease on every iteration. It may take up to 5 minutes to run 2500 iterations. Check if the "Cost after iteration 0" matches the expected output below, if not click on the square (⬛) on the upper bar of the notebook to stop the cell and try to find your error.

# In[13]:


# Congrats! It seems that your 4-layer neural network has better performance (80%) than your 2-layer neural network (72%) on the same test set. 
# 
# This is good performance for this task. Nice job! 
# 
# Though in the next course on "Improving deep neural networks" you will learn how to obtain even higher accuracy by systematically searching for better hyperparameters (learning_rate, layers_dims, num_iterations, and others you'll also learn in the next course). 

# ##  6) Results Analysis
# 
# First, let's take a look at some images the L-layer model labeled incorrectly. This will show a few mislabeled images. 

# In[16]:

# **A few types of images the model tends to do poorly on include:** 
# - Cat body in an unusual position
# - Cat appears against a background of a similar color
# - Unusual cat color and species
# - Camera Angle
# - Brightness of the picture
# - Scale variation (cat is very large or small in image) 

# 7) Test with your own image (optional/ungraded exercise)


# **References**:
# 
# - for auto-reloading external module: http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython

def saveparameters(params : dict):
    dir = "datasets/parameters/"
    for (filename, value) in params.items():
        # file name does not need to contain ".npy"
        np.save(dir+filename, value)


def loadparameters():
    dir = "datasets/parameters/"

    from os import listdir
    from os.path import isfile, join
    # no need to contain '.npy'
    temp = {}
    onlyfiles = [f for f in listdir(dir) if isfile(join(dir, f))]
    for f in onlyfiles:
        key = f.split('.')[0]
        temp[key] = np.load(dir+f)

    return temp
    

def test_ios(my_image, parameters):
	# @param my_image : just the file name
	''' NOTE: need to manually set the params cause' we can't read h5 on iOS'''
	
	classes = ["non-cat", "cat"]
	num_px = 64
	from PIL import Image, ImageFont, ImageDraw
	import tempfile
	
	totest = "images/"+my_image
	# process the file
	# the reshape will leave the ratio original
	# tempf = open('temp.tmp')
	im = Image.open(totest)
	im.load()
	
	resizedim = im.resize((num_px, num_px), Image.ANTIALIAS)
	draw = ImageDraw.Draw(resizedim)
	font = ImageFont.load_default()
	greencolor = (0,255,0)
	redcolor = (255, 0, 0)
	plt.figure()
	'''
	plt.imshow(resizedim)
	plt.show()
	plt.close()
	'''
	# get the data and reshape
	x = np.asarray(resizedim, dtype='int32').reshape((num_px * num_px * 3, 1))
	x = x/255
	# predict and show result
	my_predicted_image = predict(x, None, parameters)
	result_number = int(np.squeeze(my_predicted_image))
	if result_number==1: # meaning YES
		draw.text((5,25), "YES, A CAT", greencolor, font=font)
	else:
		draw.text((32,32), "NO!", redcolor, font=font)
		
	plt.imshow(resizedim)
	plt.show()
	plt.close()
	print("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[result_number] + "\" picture.")
	

def test(my_image, parameters):  # TODO adapt it to iOS
    global classes
    if my_image == "":
        exit()
    fname = "images/" + my_image
    # the reshape will run perfectly
    image = np.array(ndimage.imread(fname, flatten=False))
    my_image = scipy.misc.imresize(image, size=(num_px, num_px)).reshape((num_px * num_px * 3, 1))
    my_image = my_image / 255.
    my_predicted_image = predict(my_image, None, parameters)

    result_number = int(np.squeeze(my_predicted_image))
    print("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[
        result_number,].decode("utf-8") + "\" picture.")


def load_and_test():
    blue("Loading parameters ... \n")
    parameters = loadparameters()
    green("Completed\n")
    onios = False
    print('Running on iOS? (Y/y) or others? (input anything else)')
    if input().upper() == 'Y':
    	onios = True
    # totest = ["cat3.jpeg",]
    totest = justfilenames('images/')
    for i in totest:
        blue("for image: "+i)
        if not (".jpeg" in i or ".jpg" in i):
        	print("Format error: only support jpg and jpeg now")
        	continue
        if onios:
        	test_ios(i, parameters)
        else:
        	test(i, parameters)
        time.sleep(1)


def main():
    global m_train, num_px, m_test, classes
    '''
    plt.rcParams['figure.figsize'] = (5.0, 4.0)  # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'
    '''
    
    np.random.seed(1)
    blue("Loading necessary data ...")
    try:
    	train_x_orig, train_y, test_x_orig, test_y, classes = load_data_from_npy()
    	green("Completed\n")
    	index = 100
    	index = 100
    	# plt.imshow(train_x_orig[index])
    	# print ("y = " + str(train_y[0,index]) + ". It's a " + classes[train_y[0,index]].decode("utf-8") +  " picture.")

	    # Explore your datasets

    	m_train = train_x_orig.shape[0]
    	num_px = train_x_orig.shape[1]
    	m_test = test_x_orig.shape[0]
    	m_train = train_x_orig.shape[0]
    	num_px = train_x_orig.shape[1]
    	m_test = test_x_orig.shape[0]
    	
    	train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0],-1).T
        # The "-1" makes reshape flatten the remaining dimensions
    	test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

	    # Standardize data to have feature values between 0 and 1.
    	train_x = train_x_flatten / 255.
    	test_x = test_x_flatten / 255.
    except:
    	print("Reading H5 file failed")
    	print("(Skip this if you are running on iOS)")

    print("If you want to load previously trained parameters and test them, input 'Y' or 'y';")
    print("or enter anything else to retrain the model.")
    if input().upper() == "Y":
        print('\n'*100)  # clear screen
        load_and_test()
        exit()


    blue("Loading data ... \n")

    print("train_x's shape: " + str(train_x.shape))
    print("test_x's shape: " + str(test_x.shape))

    green("\nLoading completed\n")

    n_x = 12288  # num_px * num_px * 3
    n_h = 7
    n_y = 1
    layers_dims = (n_x, n_h, n_y)

    layers_dims = [12288, 20, 7, 5, 1]  # 4-layer model
    # You can train 2-layered model. Methods are already implemented.

    pass

    # Training L-layered model
    blue("Testing L-layered model")
    parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations=25000, print_cost=True)

    pred_train = predict(train_x, train_y, parameters)
    pred_test = predict(test_x, test_y, parameters)
    green("Test completed\n")
    # printing out results
    print_mislabeled_images(classes, test_x, test_y, pred_test)
    # Saving parameters to a file
    print("Do you want to replace the old parameters with current ones? (Y/y for yes")
    if input().upper()=="Y":
        saveparameters(parameters)
    else:
        green("\nMission completed.")

if __name__=="__main__":
    main()
