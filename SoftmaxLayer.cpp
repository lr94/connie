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
    size_t inputSize = input->getDataSize();

    for (unsigned i = 0; i < inputSize; i++)
    {
        float indicator = (y == i) ? 1.0f : 0.0f;
        dInput->set(i, output->get(i) - indicator);
    }

    loss = -std::log(output->get(y));
}

unsigned SoftmaxLayer::getNumClasses()
{
    return numClasses;
}

void SoftmaxLayer::setY(unsigned y)
{
    if (y >= numClasses)
        throw std::runtime_error("Invalid class");

    this->y = y;
}

float SoftmaxLayer::getLoss()
{
    return loss;
}

void SoftmaxLayer::prepend(LayerBase *previousLayer)
{
    LayerBase::prepend(previousLayer);

    delete output;
    delete dOutput;

    // The output has the same shape of the input
    output = new Vol<>(input->depth(), input->height(), input->width());
    dOutput = new Vol<>(input->depth(), input->height(), input->width());

    numClasses = static_cast<unsigned>(output->getDataSize());
}