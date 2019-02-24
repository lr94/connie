#ifndef CNN_SGDTRAINER_HPP
#define CNN_SGDTRAINER_HPP

#include "TrainerBase.hpp"

class SGDTrainer : public TrainerBase
{
public:
    explicit SGDTrainer(float learningRate) : learningRate(learningRate) { }

    float learningRate;

    void updateLayerParams(std::vector<float> &params, std::vector<float> &gradient, std::vector<float> &memory) const override;
    void updateLayerParams(Tensor<> &params, Tensor<> &gradient, Tensor<> &memory) const override;
};


#endif //CNN_SGDTRAINER_HPP
