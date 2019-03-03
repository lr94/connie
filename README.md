# Connie
Connie is a simple library for convolutional neural networks.
It was developed mainly as a personal exercise to test my understanding of artificial neural networks, so obviously don't expect something like Tensorflow...however it works and being very lightweight maybe it could even be useful in some particular cases.

## Getting started
To compile Connie using CMake:
```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```
The library itself does not depend on third party libraries, however the tests need Catch2 to be installed and some of the examples need external libraries (image_regression depends on SDL2 and libgd).

## Features
- Several nonlinearities supported (Sigmoid, Tanh, ReLU and LeakyReLU)
- Convolutional networks
- Stochastic Gradient Descent (optionally with momentum)

## License
As mentioned before the library was developed for didactic purposes, however it is released under LGPL 2.1 license
