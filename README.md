(Yoshihiro Mizoguchi 2018/07/19)

(1) Download 'MNIST original' and save them into 'dump/mnist.bin'.

    python mnist_init.py

(2) Creating a neural network using 1-epock learning,
    saving NN models to 'dump/mnist00001.h5',
    saving learning records into 'dump/mnist00001.csv',
    and saving NN model chart intu 'dump/mnist.pdf'.

    python mnist.py 1

(3) Creating a neural network using 10-epock learning
    saving NN models to 'dump/mnist00010.h5',
    saving learning records into 'dump/mnist00010.csv',
    and saving NN model chart intu 'dump/mnist.pdf'.

    python mnist.py 10

(4) Checking a nuural network made by 'dump/mnist00001.h5'
    and display learning log 'dump/mnist00001.csv' into 'dump/mnist00001.pdf'.

    python mnist_test.py 1

(5) Checking a nuural network made by 'dump/mnist00010.h5'.
    and display learning log 'dump/mnist00010.csv' into 'dump/mnist00010.pdf'.

    python mnist_test.py 10

(6) Draw data in 'MNIST original' using matplotlib (!!Under Construction!!)

    python viewdata.py

Requirements for AWS (Deep Learning AMI (Ubuntu) Version 11.0 - ami-c47c28bc)  
    pip install keras   
    pip install theano   
    pip install pydot   
