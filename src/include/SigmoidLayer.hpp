#ifndef CONNIE_SIGMOIDLAYER_HPP
#define CONNIE_SIGMOIDLAYER_HPP

#include "LayerBase.hpp"

class SigmoidLayer : public LayerBase
{
public:
    SigmoidLayer();

    ~SigmoidLayer();

    void forward() override;

    void backward() override;

protected:
    void prepend(LayerBase *previousLayer) override;
};


#endif //CONNIE_SIGMOIDLAYER_HPP
