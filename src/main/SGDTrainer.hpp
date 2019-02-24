#ifndef CNN_SGDTRAINER_HPP
#define CNN_SGDTRAINER_HPP

#include "TrainerBase.hpp"

class SGDTrainer : public TrainerBase
{
public:
    explicit SGDTrainer(Net &network, float learningRate) : TrainerBase(network), learningRate(learningRate) { }

    float learningRate;

    void train() override;
    void updateLayerParams(std::vector<float> &params, std::vector<float> &gradient) const override;
    void updateLayerParams(Tensor<> &params, Tensor<> &gradient) const override;

private:
    unsigned batchSize;
    unsigned long long i;
};


#endif //CNN_SGDTRAINER_HPP
