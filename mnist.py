#
# 'python mnist.py #epochs'
# creating a neural network using #epochs learning
#
import os, sys
os.environ['KERAS_BACKEND'] = 'theano'
import keras as ks
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import CSVLogger, TensorBoard
from keras.utils.vis_utils import plot_model
import pandas as pd
import pickle
#
if len(sys.argv)==0:
    epochs = 1
else:
    epochs = int(sys.argv[1])
filename='./dump/mnist'+"{0:05d}".format(epochs)+'.h5'
fcsv='./dump/mnist'+"{0:05d}".format(epochs)+'.csv'
#
f=open('./dump/mnist.bin','rb')
mnist=pickle.load(f)
X, y = mnist['data'], mnist['target']
X_train, y_train = X[:60000], y[:60000]
print('Training Size: ',X_train.shape)
#
X_train = X_train.reshape(X_train.shape[0], 28, 28,1)
X_train = X_train.astype('float32')
X_train /= 255
y_train = np_utils.to_categorical(y_train, 10)
#
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
fm='./dump/mnist.pdf'
print('Saving model flowchart to '+fm)
plot_model(model, to_file=fm, show_shapes=True)
#
print('Training (epochs='+str(epochs)+')')
callbacks=[]
callbacks.append(CSVLogger(fcsv))
model.fit(X_train, y_train, epochs=epochs, callbacks=callbacks)
print('Saving learning logs to '+fcsv)
print('Saveing '+filename)
model.save(filename)
#
