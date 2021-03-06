cmake_minimum_required(VERSION 3.13)
project(connie)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "-pedantic -Wall")
set(CMAKE_C_FLAGS_RELEASE "-O3")

include_directories(src/include)

# Library
add_library(connie src/include/Tensor.hpp src/include/LayerBase.hpp src/FullyConnectedLayer.cpp src/include/FullyConnectedLayer.hpp src/include/InputLayer.hpp src/include/SigmoidLayer.hpp src/include/ReluLayer.hpp src/SoftmaxLayer.cpp src/include/SoftmaxLayer.hpp src/RegressionLayer.cpp src/include/RegressionLayer.hpp src/Net.cpp src/include/Net.hpp src/include/LossLayerBase.hpp src/include/TrainerBase.hpp src/SGDTrainer.cpp src/include/SGDTrainer.hpp src/include/TanhLayer.hpp src/ReluLayer.cpp src/SigmoidLayer.cpp src/TanhLayer.cpp src/InputLayer.cpp src/LayerBase.cpp src/ConvolutionalLayer.cpp src/include/ConvolutionalLayer.hpp src/PoolingLayerBase.cpp src/include/PoolingLayerBase.hpp src/MaxPoolingLayer.cpp src/include/MaxPoolingLayer.hpp src/TrainerBase.cpp src/MomentumTrainer.cpp src/include/MomentumTrainer.hpp src/NesterovTrainer.cpp src/include/NesterovTrainer.hpp src/AdaGradTrainer.cpp src/include/AdaGradTrainer.hpp src/RMSPropTrainer.cpp src/include/RMSPropTrainer.hpp src/DropoutLayer.cpp src/include/DropoutLayer.hpp)

# Tests
find_path(CATCH2_INCLUDE_DIR NAMES catch2/catch.hpp)
if (NOT CATCH2_INCLUDE_DIR)
    message(WARNING "Catch2 testing framework not found, the tests will not be compiled")
else()
    include_directories(${CATCH2_INCLUDE_DIR})
    add_executable(tests tests/tensor.cpp tests/learning.cpp tests/MemoryStream.hpp tests/convolution.cpp tests/pooling.cpp)
    target_link_libraries(tests connie)
endif()

# XOR
add_executable(xor examples/xor/xor.cpp)
target_link_libraries(xor connie)

# Image regression
find_library(GD_LIBRARY gd)
find_library(SDL2_LIBRARY SDL2)
if(NOT GD_LIBRARY OR NOT SDL2_LIBRARY)
    message(WARNING "The image regression example needs libgd and libSDL2, which could not be found. It will not be compiled")
else()
    include_directories(${SDL2_INCLUDE_DIR} ${GD_INCLUDE_DIR}) # SDL2_INCLUDE_DIR = ?
    add_executable(image_regression examples/image_regression/image_regression.cpp)
    target_link_libraries(image_regression connie ${SDL2_LIBRARY} ${GD_LIBRARY})
endif()

# MNIST
add_executable(mnist examples/mnist/mnist.cpp examples/mnist/MnistSample.hpp examples/mnist/MnistDataset.hpp)
target_link_libraries(mnist connie stdc++fs)

# Digit classifier
if(NOT GD_LIBRARY)
    message(WARNING "The digit classifier example needs libgd, which could not be found. It will not be compiled")
else()
    include_directories(${GD_INCLUDE_DIR})
    add_executable(digit_classifier examples/digit_classifier/digit_classifier.cpp)
    target_link_libraries(digit_classifier connie ${GD_LIBRARY})
endif()

install(TARGETS connie DESTINATION lib)
install(DIRECTORY src/include/ DESTINATION include/connie FILES_MATCHING PATTERN "*.hpp")