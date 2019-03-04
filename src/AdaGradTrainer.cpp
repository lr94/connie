#include <cmath>
#include "AdaGradTrainer.hpp"

AdaGradTrainer::AdaGradTrainer(Net &network, float learningRate, unsigned batchSize)
        : AdaGradTrainer(network, learningRate, 10e-7f, batchSize) {}

AdaGradTrainer::AdaGradTrainer(Net &network, float learningRate, float delta, unsigned batchSize)
        : TrainerBase(network, batchSize, 1), learningRate(learningRate), delta(delta) {}

void AdaGradTrainer::updateLayerParams(std::vector<float> &params, std::vector<float> &gradient, std::vector<std::vector<float>> &addMem) const
{
    size_t size = params.size();

    std::vector<float> &r = addMem[0];

    for (unsigned i = 0; i < size; i++)
    {
        float g = gradient[i];
        r[i] += g * g;
        params[i] -= learningRate / (delta + std::sqrt(r[i])) * g / batchSize;
    }

    // Zero out the gradient if needed (end of minibatch)
    for (auto &g : gradient)
        g = 0;
}

void AdaGradTrainer::updateLayerParams(Tensor<> &params, Tensor<> &gradient, std::vector<Tensor<>> &addMem) const
{
    size_t size = params.getDataSize();

    Tensor<> &r = addMem[0];

    for (unsigned i = 0; i < size; i++)
    {
        float g = gradient.get(i);
        r.addAt(i, g * g);
        params.addAt(i, -learningRate / (delta + std::sqrt(r.get(i))) * g / batchSize);
    }

    // Zero out the gradient if needed (end of minibatch)
    gradient.zero();
}