# Connie
Connie is a simple library for convolutional neural networks.
It was developed mainly as a personal exercise to test my understanding of artificial neural networks, so obviously don't expect something like Tensorflow...however it works and being very lightweight maybe it could even be useful in some particular cases.

## Features  
- Several nonlinearities supported:  
  - Logistic Sigmoid  
  - Tanh  
  - ReLU  
  - Leaky ReLU  
- Convolutional networks with max pooling
- Dropout
- Several optimization methods:
  - "Vanilla" Stochastic Gradient Descent (SGD)
  - Momentum SGD
  - Nesterov
  - AdaGrad
  - RMSProp (optionally with Nesterov momentum)

## Examples
The folder `examples` contains four sample programs:
- **XOR problem** solution using a very very simple fully connected network
- **MNIST digit classification** using a network similar to LeNet-5 or a fully connected one
- **Digit classification** from an image file, using the network trained with the MNIST example
- **Image regression**: we use a deep fully connected network to solve a regression problem, with x and y the two input variables and r, g, b (i.e. the color of the pixel) the output variables.

![image](http://www.lucarobbiano.net/host/permanenti/compare_image_regression_connie_1.png)

Comparison between the original images (first row) and the images rebuilt by the network

## Build
To compile Connie using CMake:
```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```
The library itself does not depend on third party libraries, however the tests need Catch2 to be installed and some of the examples need external libraries (image_regression depends on SDL2 and libgd).
Once compiled the library can be installed with
```
sudo make install
```
The default installation prefix is /usr/local, but one can choose to install the library in a different location:
```
sudo make DESTDIR=/usr install
```
I haven't tried yet to compile Connie on Windows, but I think it should work without problems.

## License
As mentioned before the library was developed for didactic purposes, however it is released under LGPL 2.1 license
