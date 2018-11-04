import numpy as np
a = np.random.randn(3,3,3)
a = np.pad(a, ((0,0),(2,2),(2,2)), 'constant')
print(a)
