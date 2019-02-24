#include <memory>
#include <stdexcept>
#include "Net.hpp"
#include "LossLayerBase.hpp"

Net &Net::appendLayer(std::shared_ptr<LayerBase> layer)
{
    if (layers.empty())
        input = layer->input;
    else
        layers.back()->append(layer.get());

    layers.push_back(layer);
    output = layer->output;

    return *this;
}

void Net::forward()
{
    for (auto &layer : layers)
        layer->forward();
}

void Net::backward()
{
    for (auto i = layers.rbegin(); i != layers.rend(); i++)
        (*i)->backward();
}

void Net::train(TrainerBase &trainer)
{
    forward();
    backward();

    for (auto i = layers.rbegin(); i != layers.rend(); i++)
        (*i)->updateParams(trainer);
}

Tensor<> &Net::getInput()
{
    return *input;
}

Tensor<> &Net::getOutput()
{
    return *output;
}

float Net::getLoss()
{
    auto lossLayer = dynamic_cast<LossLayerBase*>(layers.back().get());

    if (lossLayer == nullptr)
        throw std::runtime_error("Cannot get loss from an incomplete network (the last layer must be a LossLayer!)");

    return lossLayer->getLoss();
}