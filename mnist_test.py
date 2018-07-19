#
# 'python mnist_test.py #no'
#  checking a nuural network made by 'mnist'+#no e.g. mnist01 or mnist10
#
import os,sys
os.environ['KERAS_BACKEND'] = 'theano'
import keras as ks
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils
from sklearn.datasets import fetch_mldata
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
#
if len(sys.argv)==0:
    epochs = 1
else:
    epochs = int(sys.argv[1])
filename='./dump/mnist'+"{0:05d}".format(epochs)+'.h5'
fcsv='./dump/mnist'+"{0:05d}".format(epochs)+'.csv'
fpdf='./dump/mnist'+"{0:05d}".format(epochs)+'.pdf'
#
df = pd.read_csv(fcsv)
data = df.values.tolist()
epoch = [x[0]+1 for x in data]
acc =  [x[1] for x in data]
loss = [x[2] for x in data]
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
p1 = ax1.plot(epoch, acc)
p2 = ax1.plot(epoch, loss)
plt.legend((p1[0], p2[0]), ("accuracy", "loss"), loc=3)
plt.title('MNIST ('+str(epochs)+' epochs learning)')
ax1.set_xlabel('epochs')
ax1.set_ylabel('rate')
ax1.set_ylim(0,1)
ax1.grid(True)
# fig.show()
fig.savefig(fpdf)
print('Saving learning logs graph to '+fpdf)
#
f=open('./dump/mnist.bin','rb')
mnist=pickle.load(f)
X, y = mnist['data'], mnist['target']
X_test, y_test = X[60000:], y[60000:]
y_test_backup = y_test
X_test = X_test.reshape(X_test.shape[0], 28, 28,1)
X_test = X_test.astype('float32')
X_test /= 255
y_test = np_utils.to_categorical(y_test, 10)
#
print('Loading model structure from '+filename)
model = load_model(filename)
#
print('Testing...')
loss_and_metrics = model.evaluate(X_test, y_test, batch_size=128)
print('Results: ',loss_and_metrics)
#
predictions = model.predict_classes(X_test)
x = list(predictions)
y = list(y_test_backup)
results = pd.DataFrame({'Actual': y, 'Predictions': x})
# print(results)

