#include "SGDTrainer.hpp"

void SGDTrainer::train()
{
    iteration++;

    net.forward();
    net.backward();

    for (auto currentLayer = layers.rbegin(); currentLayer != layers.rend(); currentLayer++)
        (*currentLayer)->updateParams(*this);
}

bool SGDTrainer::needToZeroOut() const
{
    return iteration % batchSize == 0;
}


void SGDTrainer::updateLayerParams(std::vector<float> &params, std::vector<float> &gradient) const
{
    if (iteration % batchSize != 0)
        return;

    size_t size = params.size();

    for (unsigned i = 0; i < size; i++)
        params[i] -= learningRate * gradient[i] / batchSize;
}

void SGDTrainer::updateLayerParams(Tensor<> &params, Tensor<> &gradient) const
{
    if (iteration % batchSize != 0)
        return;

    size_t size = params.getDataSize();

    for (unsigned i = 0; i < size; i++)
        params.addAt(i, -learningRate * gradient.get(i) / batchSize);
}