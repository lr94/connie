#ifndef CNN_SGDTRAINER_HPP
#define CNN_SGDTRAINER_HPP

#include "TrainerBase.hpp"

class SGDTrainer : public TrainerBase
{
public:
    explicit SGDTrainer(Net &network, float learningRate, unsigned batchSize);

    float learningRate;

    void train() override;
    bool needToZeroOut() const override;
    void updateLayerParams(std::vector<float> &params, std::vector<float> &gradient) const override;
    void updateLayerParams(Tensor<> &params, Tensor<> &gradient) const override;
    float getLoss() const;

private:
    unsigned batchSize;
    unsigned long long iteration = 0;
    float loss = 0.0f;
    float lossAccumulator = 0.0f;
};


#endif //CNN_SGDTRAINER_HPP
