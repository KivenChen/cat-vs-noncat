''' due to the Coursera version of NN is based on "h5py"
I get it to "numpy" for better compatibility
and write this to add training examples to the npy file
which contains the training set
0808-18 added deletion feature
'''

from PIL import Image 
import numpy as np
from wheels import *
import os

def add_train():
	# load both train_one and train_zero file names
	num_px = 64
	classes = ["non-cat", "cat"]
	train_one = ["train_one/"+n for n in justfilenames("train_one/")]
	train_zero = ["train_zero/"+n for n in justfilenames("train_zero/")]
	train_x = np.load('datasets/train_x_orig.npy')
	train_y = np.load('datasets/train_y.npy')
	num_x = train_x.shape[0]
	temp_x = np.array([])
	temp_y = np.array([])
		
	for i in train_one: # import train_one items
		img = Image.open(i)
		img.load()
		resizedim = img.resize((num_px, num_px), Image.ANTIALIAS)
		x = np.asarray(resizedim, dtype='int32').reshape((num_px*num_px*3, 1))
		temp_x = np.append(temp_x, x)
		temp_y = np.append(temp_y, 1)
		num_x += 1
	[os.remove(file) for file in train_one]
	for i in train_zero:
		img = Image.open(i)
		img.load()
		resizedim = img.resize((num_px, num_px), Image.ANTIALIAS)
		x = np.asarray(resizedim, dtype='int32').reshape((num_px*num_px*3, 1))
		temp_x = np.append(temp_x, x)
		temp_y = np.append(temp_y, 0)
		num_x += 1
	[os.remove(file) for file in train_zero]
		
	
	print(num_x)	
	train_x = np.append(train_x, temp_x).reshape((num_x, 64, 64, 3))
	train_y = np.append(train_y, temp_y).reshape((1, num_x))
	print(train_x.shape)
	np.save('datasets/train_x_orig.npy', train_x)
	np.save('datasets/train_y.npy', train_y)
	
	print('finished')
	print("DO remember to drop all added images.")

	
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
		draw.text((5,25), "YES,A CAT", greencolor, font=font)
	else:
		draw.text((32,32), "NO!", redcolor, font=font)
		
	plt.imshow(resizedim)
	plt.show()
	plt.close()
	print("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[result_number] + "\" picture.")

add_train()
	

	

