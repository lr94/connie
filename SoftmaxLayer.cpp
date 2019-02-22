#include <cmath>
#include <vector>
#include "SoftmaxLayer.hpp"

SoftmaxLayer::SoftmaxLayer()
{
    input = output = dInput = dOutput = nullptr;
}

SoftmaxLayer::~SoftmaxLayer()
{
    delete output;
    delete dOutput;
}

void SoftmaxLayer::forward()
{
    size_t inputSize = input->getDataSize();
    float sum = 0;

    for (unsigned i = 0; i < inputSize; i++)
    {
        float e = std::exp(input->get(i));
        output->set(i, e);
        sum += e;
    }

    (*output) /= sum;
}

void SoftmaxLayer::backward()
{

}

void SoftmaxLayer::prepend(LayerBase *previousLayer)
{
    LayerBase::prepend(previousLayer);

    delete output;
    delete dOutput;

    // The output has the same shape of the input
    output = new Vol<>(input->depth(), input->height(), input->width());
    dOutput = new Vol<>(input->depth(), input->height(), input->width());
}