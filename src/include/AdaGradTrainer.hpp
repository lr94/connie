#ifndef CONNIE_ADAGRADTRAINER_HPP
#define CONNIE_ADAGRADTRAINER_HPP

#include "TrainerBase.hpp"

class AdaGradTrainer : public TrainerBase
{
public:
    AdaGradTrainer(Net &network, float learningRate, unsigned batchSize);
    AdaGradTrainer(Net &network, float learningRate, float delta, unsigned batchSize);

    float learningRate;
    float delta;

    void updateLayerParams(std::vector<float> &params, std::vector<float> &gradient, std::vector<std::vector<float>> &addMem) const override;
    void updateLayerParams(Tensor<> &params, Tensor<> &gradient, std::vector<Tensor<>> &addMem) const override;
};


#endif //CONNIE_ADAGRADTRAINER_HPP
