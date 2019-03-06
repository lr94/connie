#ifndef CONNIE_SOFTMAXLAYER_HPP
#define CONNIE_SOFTMAXLAYER_HPP

#include "LossLayerBase.hpp"

class SoftmaxLayer : public LossLayerBase
{
public:
    SoftmaxLayer();
    ~SoftmaxLayer();

    void forward() override;
    void backward() override;

    unsigned getNumClasses();
    void setTargetClass(unsigned y);
    float getLoss() override;
    unsigned getPredictedClass();

protected:
    void prepend(LayerBase *previousLayer) override;
    unsigned numClasses;
    unsigned y;
    float loss;
    unsigned predictedClass;
};


#endif //CONNIE_SOFTMAXLAYER_HPP
