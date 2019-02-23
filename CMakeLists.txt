cmake_minimum_required(VERSION 3.13)
project(cnn)

set(CMAKE_CXX_STANDARD 14)

add_executable(cnn main.cpp Vol.hpp LayerBase.hpp FullyConnectedLayer.cpp FullyConnectedLayer.hpp InputLayer.hpp SigmoidLayer.hpp ReluLayer.hpp SoftmaxLayer.cpp SoftmaxLayer.hpp RegressionLayer.cpp RegressionLayer.hpp Net.cpp Net.hpp)