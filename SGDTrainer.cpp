#include "SGDTrainer.hpp"

void SGDTrainer::updateLayerParams(std::vector<float> &params, std::vector<float> &gradient) const
{
    size_t size = params.size();

    for (unsigned i = 0; i < size; i++)
        params[i] -= learningRate * gradient[i];
}

void SGDTrainer::updateLayerParams(Tensor<> &params, Tensor<> &gradient) const
{
    size_t size = params.size();

    for (unsigned i = 0; i < size; i++)
        params.addAt(i, -learningRate * gradient.get(i));
}