#include <memory>
#include <stdexcept>
#include <fstream>
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

bool Net::save(const char *filename)
{
    std::ofstream outputStream(filename, std::ofstream::binary);
    bool ret = save(outputStream);
    outputStream.close();
    return ret;
}

bool Net::load(const char *filename)
{
    std::ifstream inputStream(filename, std::ifstream::binary);
    bool ret = load(inputStream);
    inputStream.close();
    return ret;
}

bool Net::save(std::ostream &stream)
{
    for (auto &layer : layers)
        if (!layer->save(stream))
            return false;

    return true;
}

bool Net::load(std::istream &stream)
{
    for (auto &layer : layers)
        if (!layer->load(stream))
            return false;

    return true;
}