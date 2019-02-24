#include "SGDTrainer.hpp"

void SGDTrainer::train()
{
    net.forward();
    net.backward();

    for (auto i = layers.rbegin(); i != layers.rend(); i++)
        (*i)->updateParams(*this);
}

void SGDTrainer::updateLayerParams(std::vector<float> &params, std::vector<float> &gradient, std::vector<float> &memory) const
{
    size_t size = params.size();

    for (unsigned i = 0; i < size; i++)
        params[i] -= learningRate * gradient[i];
}

void SGDTrainer::updateLayerParams(Tensor<> &params, Tensor<> &gradient, Tensor<> &memory) const
{
    size_t size = params.getDataSize();

    for (unsigned i = 0; i < size; i++)
        params.addAt(i, -learningRate * gradient.get(i));
}