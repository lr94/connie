#ifndef CNN_SGDTRAINER_HPP
#define CNN_SGDTRAINER_HPP

#include "TrainerBase.hpp"

/**
 * Plain Stochastic Gradient Descent implementation
 */
class SGDTrainer : public TrainerBase
{
public:
    SGDTrainer(Net &network, float learningRate, unsigned batchSize);

    float learningRate;

    void updateLayerParams(std::vector<float> &params, std::vector<float> &gradient, std::vector<std::vector<float>> &addMem) const override;
    void updateLayerParams(Tensor<> &params, Tensor<> &gradient, std::vector<Tensor<>> &addMem) const override;
};


#endif //CNN_SGDTRAINER_HPP
