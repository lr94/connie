#include <cmath>
#include "Tensor.hpp"
#include "ReluLayer.hpp"

static inline float relu(float x)
{
    return std::fmaxf(0, x);
}

static inline float relu_derivative(float x)
{
    if (x <= 0.0f)
        return 0.0f;
    else
        return 1.0f;
}

ReluLayer::ReluLayer()
{
    input = output = dInput = dOutput = nullptr;
}

ReluLayer::~ReluLayer()
{
    delete output;
    delete dOutput;
}

void ReluLayer::forward()
{
    size_t inputSize = input->getDataSize();

    for (unsigned i = 0; i < inputSize; i++)
        output->set(i, relu(input->get(i)));
}

void ReluLayer::backward()
{
    size_t inputSize = input->getDataSize();

    for (unsigned i = 0; i < inputSize; i++)
        dInput->set(i, dOutput->get(i) * relu_derivative(output->get(i)));
}

void ReluLayer::prepend(LayerBase *previousLayer)
{
    LayerBase::prepend(previousLayer);

    delete output;
    delete dOutput;

    // The output has the same shape of the input
    output = new Tensor<>(input->depth(), input->height(), input->width());
    dOutput = new Tensor<>(input->depth(), input->height(), input->width());
}