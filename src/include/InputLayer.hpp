#ifndef CONNIE_INPUTLAYER_HPP
#define CONNIE_INPUTLAYER_HPP

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

#endif //CONNIE_INPUTLAYER_HPP
