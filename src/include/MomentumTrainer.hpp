#ifndef CONNIE_MOMENTUMTRAINER_HPP
#define CONNIE_MOMENTUMTRAINER_HPP

#include "TrainerBase.hpp"

class MomentumTrainer : public TrainerBase
{
public:
    MomentumTrainer(Net &network, float learningRate, float momentum, unsigned batchSize);

    float learningRate;
    float momentum;

    void updateLayerParams(std::vector<float> &params, std::vector<float> &gradient, std::vector<std::vector<float>> &addMem) const override;
    void updateLayerParams(Tensor<> &params, Tensor<> &gradient, std::vector<Tensor<>> &addMem) const override;
};


#endif //CONNIE_MOMENTUMTRAINER_HPP
