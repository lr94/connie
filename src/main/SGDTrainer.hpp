#ifndef CNN_SGDTRAINER_HPP
#define CNN_SGDTRAINER_HPP

#include "TrainerBase.hpp"

class SGDTrainer : public TrainerBase
{
public:
    explicit SGDTrainer(Net &network, float learningRate, unsigned batchSize) : TrainerBase(network), learningRate(learningRate), batchSize(batchSize), iteration(0)
    {
        if (batchSize == 0)
            throw std::runtime_error("Invalid batch size");
    }

    float learningRate;

    void train() override;
    bool needToZeroOut() const override;
    void updateLayerParams(std::vector<float> &params, std::vector<float> &gradient) const override;
    void updateLayerParams(Tensor<> &params, Tensor<> &gradient) const override;

private:
    unsigned batchSize;
    unsigned long long iteration;
};


#endif //CNN_SGDTRAINER_HPP
