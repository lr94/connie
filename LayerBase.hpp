#ifndef CNN_LAYER_HPP
#define CNN_LAYER_HPP

#include "Vol.hpp"

/**
 * Each layer is responsible of its output tensor, the input one is the output tensor of the previous layer
 */
class LayerBase
{
public:
    Vol<> *input;
    Vol<> *output;

    Vol<> *dInput;
    Vol<> *dOutput;

    virtual void forward() = 0;
    virtual void backward() = 0;

    void append(LayerBase *nextLayer)
    {
        nextLayer->prepend(this);
    }

protected:
    virtual void prepend(LayerBase *previousLayer)
    {
        input = previousLayer->output;
        dInput = previousLayer->dOutput;
    }
};
#endif
