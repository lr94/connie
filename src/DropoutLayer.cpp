#include "DropoutLayer.hpp"

DropoutLayer::DropoutLayer() : DropoutLayer(0.5f) {}

DropoutLayer::DropoutLayer(float keepProb) : p(keepProb), randomEngine(std::random_device()()), bernoulli(keepProb)
{
    input = output = dInput = dOutput = nullptr;
    mask = nullptr;
}

DropoutLayer::~DropoutLayer()
{
    delete output;
    delete dOutput;
    delete mask;
}

void DropoutLayer::forward()
{
    size_t size = input->getDataSize();

    if (trainingMode)
        for (unsigned i = 0; i < size; i++)
        {
            bool keep = bernoulli(randomEngine);
            mask->set(i, keep);
            output->set(i, keep ? input->get(i) / p : 0.0f);
        }
    else
        for (unsigned i = 0; i < size; i++)
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