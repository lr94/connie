#include <cmath>
#include "TanhLayer.hpp"

TanhLayer::TanhLayer()
{
    input = output = dInput = dOutput = nullptr;
}

TanhLayer::~TanhLayer()
{
    delete output;
    delete dOutput;
}

void TanhLayer::forward()
{
    size_t inputSize = input->getDataSize();

    for (unsigned i = 0; i < inputSize; i++)
        output->set(i, std::tanh(input->get(i)));
}

void TanhLayer::backward()
{
    size_t inputSize = input->getDataSize();

    for (unsigned i = 0; i < inputSize; i++)
    {
        float v = output->get(i);
        dInput->set(i, dOutput->get(i) * (1.0f - v * v));
    }
}

void TanhLayer::prepend(LayerBase *previousLayer)
{
    LayerBase::prepend(previousLayer);

    delete output;
    delete dOutput;

    // The output has the same shape of the input
    output = new Tensor<>(input->depth(), input->height(), input->width());
    dOutput = new Tensor<>(input->depth(), input->height(), input->width());
}