#include "SGDTrainer.hpp"

void SGDTrainer::train()
{
    i++;

    net.forward();
    net.backward();

    for (auto currentLayer = layers.rbegin(); currentLayer != layers.rend(); currentLayer++)
        (*currentLayer)->updateParams(*this);
}

bool SGDTrainer::needToZeroOut() const
{
    return i % batchSize == 0;
}


void SGDTrainer::updateLayerParams(std::vector<float> &params, std::vector<float> &gradient) const
{
    size_t size = params.size();

    for (unsigned i = 0; i < size; i++)
        params[i] -= learningRate * gradient[i];
}

void SGDTrainer::updateLayerParams(Tensor<> &params, Tensor<> &gradient) const
{
    size_t size = params.getDataSize();

    for (unsigned i = 0; i < size; i++)
        params.addAt(i, -learningRate * gradient.get(i));
}