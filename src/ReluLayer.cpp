#include <cmath>
#include <stdexcept>
#include "Tensor.hpp"
#include "ReluLayer.hpp"

static inline float relu(float x, float leak)
{
    return std::fmaxf(leak * x, x);
}

static inline float relu_derivative(float x, float leak)
{
    if (x <= 0.0f)
        return leak;
    else
        return 1.0f;
}

ReluLayer::ReluLayer() : ReluLayer(0.0f) {}

ReluLayer::ReluLayer(float leak) : leak(leak)
{
    if (leak > 1.0f || leak < 0.0f)
        throw std::runtime_error("Leaky ReLU must have a leak constant between 0 and 1");

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
        output->set(i, relu(input->get(i), leak));
}

void ReluLayer::backward()
{
    size_t inputSize = input->getDataSize();

    for (unsigned i = 0; i < inputSize; i++)
        dInput->set(i, dOutput->get(i) * relu_derivative(output->get(i), leak));
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