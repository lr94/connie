
#ifndef CNN_SOFTMAXLAYER_HPP
#define CNN_SOFTMAXLAYER_HPP

#include "LayerBase.hpp"

class SoftmaxLayer : public LayerBase
{
public:
    SoftmaxLayer();
    ~SoftmaxLayer();

    void forward() override;
    void backward() override;

    unsigned getNumClasses();
    void setY(unsigned y);
    float getLoss();

protected:
    void prepend(LayerBase *previousLayer);
    unsigned numClasses;
    unsigned y;
    float loss;
};


#endif //CNN_SOFTMAXLAYER_HPP
