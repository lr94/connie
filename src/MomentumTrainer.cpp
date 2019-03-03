#include "MomentumTrainer.hpp"

MomentumTrainer::MomentumTrainer(Net &network, float learningRate, float momentum, unsigned batchSize)
    : TrainerBase(network, 1), learningRate(learningRate), momentum(momentum)
{
    this->batchSize = batchSize;

    if (batchSize == 0)
        throw std::runtime_error("Invalid batch size");
}

void MomentumTrainer::updateLayerParams(std::vector<float> &params, std::vector<float> &gradient, std::vector<std::vector<float>> &addMem) const
{
    if (iteration % batchSize != 0)
        return;

    size_t size = params.size();

    std::vector<float> &v = addMem[0];

    for (unsigned i = 0; i < size; i++)
    {
        v[i] = momentum * v[i] - learningRate * gradient[i] / batchSize;
        params[i] += v[i];
    }

    // Zero out the gradient if needed (end of minibatch)
    for (auto &g : gradient)
        g = 0;
}

void MomentumTrainer::updateLayerParams(Tensor<> &params, Tensor<> &gradient, std::vector<Tensor<>> &addMem) const
{
    if (iteration % batchSize != 0)
        return;

    size_t size = params.getDataSize();

    Tensor<> &v = addMem[0];

    for (unsigned i = 0; i < size; i++)
    {
        v.set(i, momentum * v.get(i) - learningRate * gradient.get(i) / batchSize);
        params.addAt(i, v[i]);
    }

    // Zero out the gradient if needed (end of minibatch)
    gradient.zero();
}

float MomentumTrainer::getLoss() const
{
    return loss;
}