#ifndef CNN_SIGMOIDLAYER_HPP
#define CNN_SIGMOIDLAYER_HPP

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


#endif //CNN_SIGMOIDLAYER_HPP
