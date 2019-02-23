#ifndef CNN_SGDTRAINER_HPP
#define CNN_SGDTRAINER_HPP

#include "TrainerBase.hpp"

class SGDTrainer : public TrainerBase
{
public:
    explicit SGDTrainer(float learningRate) : learningRate(learningRate) { }

    float learningRate;

    void changeParams(std::vector<float> &params, std::vector<float> &gradient) const override;
    void changeParams(Vol<> &params, Vol<> &gradient) const override;
};


#endif //CNN_SGDTRAINER_HPP
