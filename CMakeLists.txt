cmake_minimum_required(VERSION 3.13)
project(cnn)

set(CMAKE_CXX_STANDARD 14)

include_directories(src)

add_library(cnn src/Tensor.hpp src/LayerBase.hpp src/FullyConnectedLayer.cpp src/FullyConnectedLayer.hpp src/InputLayer.hpp src/SigmoidLayer.hpp src/ReluLayer.hpp src/SoftmaxLayer.cpp src/SoftmaxLayer.hpp src/RegressionLayer.cpp src/RegressionLayer.hpp src/Net.cpp src/Net.hpp src/LossLayerBase.hpp src/TrainerBase.hpp src/SGDTrainer.cpp src/SGDTrainer.hpp src/TanhLayer.hpp)

# Main
add_executable(main_test examples/main_test.cpp)
target_link_libraries(main_test cnn)

# Image regression
find_library(GD_LIBRARY gd)
find_library(SDL2_LIBRARY SDL2)
include_directories(${SDL2_INCLUDE_DIR} ${GD_INCLUDE_DIR}) # SDL2_INCLUDE_DIR = ?
add_executable(paint_picture examples/paint_picture.cpp)
target_link_libraries(paint_picture cnn ${SDL2_LIBRARY} ${GD_LIBRARY})
