#include "SGDTrainer.hpp"

SGDTrainer::SGDTrainer(Net &network, float learningRate, unsigned batchSize) : TrainerBase(network), learningRate(learningRate)
{
    this->batchSize = batchSize;

    if (batchSize == 0)
        throw std::runtime_error("Invalid batch size");
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

float SGDTrainer::getLoss() const
{
    return loss;
}