cmake_minimum_required(VERSION 3.13)
project(cnn)

set(CMAKE_CXX_STANDARD 14)

include_directories(src/main)

add_executable(main_test src/test/main_test.cpp src/main/Tensor.hpp src/main/LayerBase.hpp src/main/FullyConnectedLayer.cpp src/main/FullyConnectedLayer.hpp src/main/InputLayer.hpp src/main/SigmoidLayer.hpp src/main/ReluLayer.hpp src/main/SoftmaxLayer.cpp src/main/SoftmaxLayer.hpp src/main/RegressionLayer.cpp src/main/RegressionLayer.hpp src/main/Net.cpp src/main/Net.hpp src/main/LossLayerBase.hpp src/main/TrainerBase.hpp src/main/SGDTrainer.cpp src/main/SGDTrainer.hpp src/main/TanhLayer.hpp)

find_library(GD_LIBRARY gd)
find_library(SDL2_LIBRARY SDL2)

include_directories(${SDL2_INCLUDE_DIR} ${GD_INCLUDE_DIR}) # SDL2_INCLUDE_DIR = ?
add_executable(paint_picture src/test/paint_picture.cpp src/main/Tensor.hpp src/main/LayerBase.hpp src/main/FullyConnectedLayer.cpp src/main/FullyConnectedLayer.hpp src/main/InputLayer.hpp src/main/SigmoidLayer.hpp src/main/ReluLayer.hpp src/main/SoftmaxLayer.cpp src/main/SoftmaxLayer.hpp src/main/RegressionLayer.cpp src/main/RegressionLayer.hpp src/main/Net.cpp src/main/Net.hpp src/main/LossLayerBase.hpp src/main/TrainerBase.hpp src/main/SGDTrainer.cpp src/main/SGDTrainer.hpp src/main/TanhLayer.hpp)
target_link_libraries(paint_picture ${SDL2_LIBRARY} ${GD_LIBRARY})
