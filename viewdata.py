#
# 'viewdata.py'
#  draw data in 'MNIST original'
#
import os
os.environ['KERAS_BACKEND'] = 'theano'
import keras as ks
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
#
f=open('./dump/mnist.bin','rb')
mnist=pickle.load(f)
X, y = mnist['data'], mnist['target']
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
y_test_backup = y_test
print('Training Size / Test Size')
print(X_train.shape,X_test.shape)
#
X_train = X_train.reshape(X_train.shape[0], 28, 28,1)
X_test = X_test.reshape(X_test.shape[0], 28, 28,1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)
#
print('X.shape and y.shape')
print(X.shape,y.shape)
#
# pick up a 53238th number as a test_number
test_number = X[53238]
print('test_number = X[53238]')
# print('X[53238]=',test_number)
#
# reshape it into 28x28(=784)
test_number_image = test_number.reshape(28,28)
pd.options.display.max_columns = 28
pd.set_option('display.width', 150)
# From numpy array to Panas dataflame
number_matrix = pd.DataFrame(test_number_image)
# print number_matrix into terminal
print(number_matrix)
# draw number_matrix using matplotlib
plt.imshow(test_number_image, cmap = matplotlib.cm.binary,interpolation='nearest')
plt.show()
