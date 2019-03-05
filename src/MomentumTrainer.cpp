#include "MomentumTrainer.hpp"

MomentumTrainer::MomentumTrainer(Net &network, float learningRate, float momentum, unsigned batchSize)
    : TrainerBase(network, batchSize, 1), learningRate(learningRate), momentum(momentum) {}

void MomentumTrainer::updateLayerParams(std::vector<float> &params, std::vector<float> &gradient, std::vector<std::vector<float>> &addMem) const
{
    size_t size = params.size();

    std::vector<float> &v = addMem[0];

    for (unsigned i = 0; i < size; i++)
    {
        float vi = momentum * v[i] - learningRate * gradient[i] / batchSize;
        v[i] = vi;
        params[i] += vi;
    }

    // Zero out the gradient (end of minibatch)
    for (auto &g : gradient)
        g = 0;
}

void MomentumTrainer::updateLayerParams(Tensor<> &params, Tensor<> &gradient, std::vector<Tensor<>> &addMem) const
{
    size_t size = params.getDataSize();

    Tensor<> &v = addMem[0];

    for (unsigned i = 0; i < size; i++)
    {
        float vi = momentum * v.get(i) - learningRate * gradient.get(i) / batchSize;
        v.set(i, vi);
        params.addAt(i, vi);
    }

    // Zero out the gradient (end of minibatch)
    gradient.zero();
}