
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

protected:
    void prepend(LayerBase *previousLayer);
};


#endif //CNN_SOFTMAXLAYER_HPP
