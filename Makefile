#
all:
	@echo "try 'make test10'"
dump/mnist.bin:
	python mnist_init.py
dump/mnist00001.h5: dump/mnist.bin
	python mnist.py 1
dump/mnist00010.h5: dump/mnist.bin
	python mnist.py 10
test01: dump/mnist00001.h5
	python mnist_test.py 1
test10: dump/mnist00010.h5
	python mnist_test.py 10
	open dump/mnist00010.pdf
view: dump/mnist.bin
	python viewdata.py
clean:
	rm -rf dump/* *~


