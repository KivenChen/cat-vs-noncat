'''
the file tested PIL.Image.thumbnail(size: tuple, option),
which is a substitude of scipy.ndimage
'''
import tempfile
from PIL import Image as img
from matplotlib import pyplot as plt
from time import sleep
import numpy as np
target = img.open('images/cat3.jpeg')
plt.figure()
plt.imshow(target)
plt.show()
tf = tempfile.NamedTemporaryFile()
print("created tempfile with name "+tf.name)
def toarray():
	target.open(tf)
	target.load()
	data = np.asarray(target, dtype='int32')
	print(data.shape)
	print(data)
''' 
NOTE: to call image.save() requires a file name instead of a file object
so we use NamedTemporaryFile()
'''	

target.thumbnail((64, 64), img.ANTIALIAS)
target.save(tf.name) 
plt.figure()
plt.imshow(target)
plt.show()

toarray()
tf.close()


