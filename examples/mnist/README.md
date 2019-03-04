# MNIST
MNIST is a dataset of 70.000 handwritten digits (60.000 are used for training, while the other 10.000 are used as test set).
The compressed dataset can be downloaded from here: http://yann.lecun.com/exdb/mnist/

In this example we use Connie to classify the digits. The program can use two different neural networks:

- Fully Connected (3 layers containing 1568, 784 and 10 units)
- A modified version of the famous LeNet-5 convolutional network. Since the original LeNet-5 network had 32x32 pixel images but our dataset contains 28x28 images we use a padding of 2. We also use max pooling instead of average pooling (not yet supported by Connie) and ReLU as activation function instead of Tanh.

## How to run the demo
Examples:

```./mnist --action train --network fc --data-file train-images-idx3-ubyte --label-file train-labels-idx1-ubyte --learning-rate 0.001```

```./mnist --action train --network lenet --data-file train-images-idx3-ubyte --label-file train-labels-idx1-ubyte --trainer rmsprop --learning-rate 0.001 --momentum 0.8 --decay-rate 0.75```

These two commands will train a network (which will be stored in ```network.bin```) using two different optimization algorithms (vanilla SGD and RMSProp)

To test the results:

```./mnist --action test --data-file t10k-images-idx3-ubyte --label-file t10k-labels-idx1-ubyte --sample-size 10000```

This command will load the LeNet-5 network stored in ```network.bin``` and test its accuracy.
The input file can be specified with the optoin ```--network-file```

If called without arguments the program will print the list of supported options.