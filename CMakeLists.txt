cmake_minimum_required(VERSION 3.13)
project(cnn)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "-pedantic -Wall")
set(CMAKE_C_FLAGS_RELEASE "-O3")

include_directories(src/include)

# Library
add_library(cnn src/include/Tensor.hpp src/include/LayerBase.hpp src/FullyConnectedLayer.cpp src/include/FullyConnectedLayer.hpp src/include/InputLayer.hpp src/include/SigmoidLayer.hpp src/include/ReluLayer.hpp src/SoftmaxLayer.cpp src/include/SoftmaxLayer.hpp src/RegressionLayer.cpp src/include/RegressionLayer.hpp src/Net.cpp src/include/Net.hpp src/include/LossLayerBase.hpp src/include/TrainerBase.hpp src/SGDTrainer.cpp src/include/SGDTrainer.hpp src/include/TanhLayer.hpp src/ReluLayer.cpp)

# Tests
find_path(CATCH2_INCLUDE_DIR NAMES catch2/catch.hpp)
if (NOT CATCH2_INCLUDE_DIR)
    message(WARNING "Catch2 testing framework not found, the tests will not be compiled")
else()
    include_directories(${CATCH2_INCLUDE_DIR})
    add_executable(tests tests/tensor.cpp tests/learning.cpp tests/MemoryStream.hpp)
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