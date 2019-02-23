cmake_minimum_required(VERSION 3.13)
project(cnn)

set(CMAKE_CXX_STANDARD 14)

add_executable(cnn src/test/main.cpp src/main/Tensor.hpp src/main/LayerBase.hpp src/main/FullyConnectedLayer.cpp src/main/FullyConnectedLayer.hpp src/main/InputLayer.hpp src/main/SigmoidLayer.hpp src/main/ReluLayer.hpp src/main/SoftmaxLayer.cpp src/main/SoftmaxLayer.hpp src/main/RegressionLayer.cpp src/main/RegressionLayer.hpp src/main/Net.cpp src/main/Net.hpp src/main/LossLayerBase.hpp src/main/TrainerBase.hpp src/main/SGDTrainer.cpp src/main/SGDTrainer.hpp)