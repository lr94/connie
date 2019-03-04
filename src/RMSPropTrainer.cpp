#include <cmath>
#include "RMSPropTrainer.hpp"

RMSPropTrainer::RMSPropTrainer(Net &network, float learningRate, float decayRate, float momentum, float delta,
        unsigned batchSize) : TrainerBase(network, batchSize, 2), learningRate(learningRate), decayRate(decayRate),
        momentum(momentum), delta(delta) {}

RMSPropTrainer::RMSPropTrainer(Net &network, float learningRate, float decayRate, unsigned batchSize)
        : RMSPropTrainer(network, learningRate, decayRate, 0.0f, 10e-6f, batchSize) {}

RMSPropTrainer::RMSPropTrainer(Net &network, float learningRate, float decayRate, float momentum, unsigned batchSize)
        : RMSPropTrainer(network, learningRate, decayRate, momentum, 10e-6f, batchSize) {}

void RMSPropTrainer::updateLayerParams(std::vector<float> &params, std::vector<float> &gradient, std::vector<std::vector<float>> &addMem) const
{
    size_t size = params.size();

    std::vector<float> &v = addMem[0];
    std::vector<float> &r = addMem[1];

    for (unsigned i = 0; i < size; i++)
    {
        float gi = gradient[i];
        float ri = decayRate * r[i] + (1 - decayRate) * gi * gi;
        r[i] = ri;
        float vi = momentum * v[i] - learningRate / (delta + std::sqrt(ri)) * gi / batchSize;
        v[i] = vi;
        params[i] += vi;
    }

    // Zero out the gradient if needed (end of minibatch)
    for (auto &g : gradient)
        g = 0;
}

void RMSPropTrainer::updateLayerParams(Tensor<> &params, Tensor<> &gradient, std::vector<Tensor<>> &addMem) const
{
    size_t size = params.getDataSize();

    Tensor<> &v = addMem[0];
    Tensor<> &r = addMem[1];

    for (unsigned i = 0; i < size; i++)
    {
        float gi = gradient.get(i);
        float ri = decayRate * r.get(i) + (1 - decayRate) * gi * gi;
        r.set(i, ri);
        float vi = momentum * v.get(i) - learningRate / (delta + std::sqrt(ri)) * gi / batchSize;
        v.set(i, vi);
        params.addAt(i, vi);
    }

    // Zero out the gradient if needed (end of minibatch)
    gradient.zero();
}

float RMSPropTrainer::getLoss() const
{
    return loss;
}