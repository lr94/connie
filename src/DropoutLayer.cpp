#include "DropoutLayer.hpp"

DropoutLayer::DropoutLayer() : DropoutLayer(0.5f) {}

DropoutLayer::DropoutLayer(float p) : p(p), randomEngine(std::random_device()()), bernoulli(p)
{
    input = output = dInput = dOutput = nullptr;
    mask = nullptr;
}

DropoutLayer::~DropoutLayer()
{
    delete input;
    delete dInput;
    delete output;
    delete dOutput;
    delete mask;
}

void DropoutLayer::forward()
{
    size_t size = input->getDataSize();

    if (trainingMode)
    {
        // Randomly set mask tensor
        for (unsigned i = 0; i < size; i++)
            mask->set(i, !bernoulli(randomEngine));
    }

    for (unsigned i = 0; i < size; i++)
        if (mask->get(i) || !trainingMode)
            output->set(i, input->get(i));
}

void DropoutLayer::backward()
{
    size_t size = input->getDataSize();

    for (unsigned i = 0; i < size; i++)
        dInput->set(i, mask->get(i) ? dOutput->get(i) : 0.0f);
}

void DropoutLayer::prepend(LayerBase *previousLayer)
{
    LayerBase::prepend(previousLayer);

    delete output;
    delete dOutput;
    delete mask;

    // The output has the same shape of the input
    output = new Tensor<>(input->depth(), input->height(), input->width());
    dOutput = new Tensor<>(input->depth(), input->height(), input->width());
    mask = new Tensor<bool>(input->depth(), input->height(), input->width());
}