#include "Net.hpp"

Net &Net::appendLayer(LayerBase *layer)
{
    if (layers.empty())
        input = layer->input;
    else
        layers.back()->append(layer);

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

Vol<> &Net::getInput()
{
    return *input;
}

Vol<> &Net::getOutput()
{
    return *output;
}