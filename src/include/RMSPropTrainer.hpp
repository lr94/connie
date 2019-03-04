#ifndef CONNIE_RMSPROPTRAINER_HPP
#define CONNIE_RMSPROPTRAINER_HPP

#include "TrainerBase.hpp"

class RMSPropTrainer : public TrainerBase
{
public:
    RMSPropTrainer(Net &network, float learningRate, float decayRate, unsigned batchSize);
    RMSPropTrainer(Net &network, float learningRate, float decayRate, float momentum, unsigned batchSize);
    RMSPropTrainer(Net &network, float learningRate, float decayRate, float momentum, float delta, unsigned batchSize);

    float learningRate;
    float decayRate;
    float momentum;
    float delta;

    void updateLayerParams(std::vector<float> &params, std::vector<float> &gradient, std::vector<std::vector<float>> &addMem) const override;
    void updateLayerParams(Tensor<> &params, Tensor<> &gradient, std::vector<Tensor<>> &addMem) const override;
};


#endif //CONNIE_RMSPROPTRAINER_HPP
