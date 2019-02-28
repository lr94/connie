cmake_minimum_required(VERSION 3.13)
project(cnn)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "-pedantic -Wall")
set(CMAKE_C_FLAGS_RELEASE "-O3")

include_directories(src)

# Library
add_library(cnn src/Tensor.hpp src/LayerBase.hpp src/FullyConnectedLayer.cpp src/FullyConnectedLayer.hpp src/InputLayer.hpp src/SigmoidLayer.hpp src/ReluLayer.hpp src/SoftmaxLayer.cpp src/SoftmaxLayer.hpp src/RegressionLayer.cpp src/RegressionLayer.hpp src/Net.cpp src/Net.hpp src/LossLayerBase.hpp src/TrainerBase.hpp src/SGDTrainer.cpp src/SGDTrainer.hpp src/TanhLayer.hpp)

# Tests
find_path(CATCH2_INCLUDE_DIR NAMES catch2/catch.hpp)
if (NOT CATCH2_INCLUDE_DIR)
    message(WARNING "Catch2 testing framework not found, the tests will not be compiled")
else()
    add_executable(tests tests/tensor.cpp)
    target_link_libraries(tests cnn)
endif()

# Main
add_executable(main_test examples/main_test/main_test.cpp)
target_link_libraries(main_test cnn)

# Image regression
find_library(GD_LIBRARY gd)
find_library(SDL2_LIBRARY SDL2)
if(NOT GD_LIBRARY OR NOT SDL2_LIBRARY)
    message(WARNING "The image regression example needs libgd and libSDL2, which could not be found. It will not be compiled")
else()
    include_directories(${SDL2_INCLUDE_DIR} ${GD_INCLUDE_DIR}) # SDL2_INCLUDE_DIR = ?
    add_executable(image_regression examples/image_regression/image_regression.cpp)
    target_link_libraries(image_regression cnn ${SDL2_LIBRARY} ${GD_LIBRARY})
endif()

# MNIST
add_executable(mnist examples/mnist/mnist.cpp examples/mnist/Sample.hpp examples/mnist/Dataset.hpp)
target_link_libraries(mnist cnn stdc++fs)