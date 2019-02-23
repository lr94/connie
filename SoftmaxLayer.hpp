#ifndef CNN_SOFTMAXLAYER_HPP
#define CNN_SOFTMAXLAYER_HPP

#include "LossLayerBase.hpp"

class SoftmaxLayer : public LossLayerBase
{
public:
    SoftmaxLayer();
    ~SoftmaxLayer();

    void forward() override;
    void backward() override;

    void updateParams(const TrainerBase &trainer) override {}

    unsigned getNumClasses();
    void setY(unsigned y);
    float getLoss() override;

protected:
    void prepend(LayerBase *previousLayer) override;
    unsigned numClasses;
    unsigned y;
    float loss;
};


#endif //CNN_SOFTMAXLAYER_HPP
