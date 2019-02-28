#include <stdexcept>
#include "Tensor.hpp"
#include "InputLayer.hpp"

InputLayer::InputLayer(unsigned depth, unsigned height, unsigned width)
{
    // In the input layer the input tensor and the output tensor are the same
    input = new Tensor<>(depth, height, width);
    dInput = new Tensor<>(depth, height, width);
    output = input;
    dOutput = dInput;
}

InputLayer::~InputLayer()
{
    delete input;
    delete dInput;
}

void InputLayer::prepend(LayerBase *previousLayer)
{
    // LayerBase::prepend(previousLayer);
    throw std::runtime_error("Cannot append an InputLayer");
}