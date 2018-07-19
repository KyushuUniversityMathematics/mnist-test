#
# 'mnist_init.py'
# download 'MNIST original' and save it to 'mnist.bin'
#
from sklearn.datasets import fetch_mldata
import pickle
mnist = fetch_mldata('MNIST original')
f=open('./dump/mnist.bin','wb')
pickle.dump(mnist,f)
