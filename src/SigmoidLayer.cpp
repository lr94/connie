#include <cmath>
#include "Tensor.hpp"
#include "SigmoidLayer.hpp"

static inline float sigmoid(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}

SigmoidLayer::SigmoidLayer()
{
    input = output = dInput = dOutput = nullptr;
}

SigmoidLayer::~SigmoidLayer()
{
    delete output;
    delete dOutput;
}

void SigmoidLayer::forward()
{
    size_t inputSize = input->getDataSize();

    for (unsigned i = 0; i < inputSize; i++)
        output->set(i, sigmoid(input->get(i)));
}

void SigmoidLayer::backward()
{
    size_t inputSize = input->getDataSize();

    for (unsigned i = 0; i < inputSize; i++)
        dInput->set(i, dOutput->get(i) * (1.0f - output->get(i)) * output->get(i));
}

void SigmoidLayer::prepend(LayerBase *previousLayer)
{
    LayerBase::prepend(previousLayer);

    delete output;
    delete dOutput;

    // The output has the same shape of the input
    output = new Tensor<>(input->depth(), input->height(), input->width());
    dOutput = new Tensor<>(input->depth(), input->height(), input->width());
}
