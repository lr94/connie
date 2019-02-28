#ifndef CNN_INPUTLAYER_HPP
#define CNN_INPUTLAYER_HPP

#include "LayerBase.hpp"

class InputLayer : public LayerBase
{
public:
    InputLayer(unsigned depth, unsigned height, unsigned width);

    ~InputLayer();

    void forward() override {}
    void backward() override {}

protected:
    void prepend(LayerBase *previousLayer) override;
};

#endif //CNN_INPUTLAYER_HPP
