#include "SGDTrainer.hpp"

SGDTrainer::SGDTrainer(Net &network, float learningRate, unsigned batchSize) : TrainerBase(network), learningRate(learningRate)
{
    this->batchSize = batchSize;

    if (batchSize == 0)
        throw std::runtime_error("Invalid batch size");
}

void SGDTrainer::updateLayerParams(std::vector<float> &params, std::vector<float> &gradient) const
{
    if (iteration % batchSize != 0)
        return;

    size_t size = params.size();

    for (unsigned i = 0; i < size; i++)
        params[i] -= learningRate * gradient[i] / batchSize;

    // Zero out the gradient if needed (end of minibatch)
    for (auto &g : gradient)
        g = 0;
}

void SGDTrainer::updateLayerParams(Tensor<> &params, Tensor<> &gradient) const
{
    if (iteration % batchSize != 0)
        return;

    size_t size = params.getDataSize();

    for (unsigned i = 0; i < size; i++)
        params.addAt(i, -learningRate * gradient.get(i) / batchSize);

    // Zero out the gradient if needed (end of minibatch)
    gradient.zero();
}

float SGDTrainer::getLoss() const
{
    return loss;
}