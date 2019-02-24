#ifndef CNN_LAYER_HPP
#define CNN_LAYER_HPP

#include "Tensor.hpp"
class TrainerBase;

/**
 * Each layer is responsible of its output tensor, the input one is the output tensor of the previous layer
 */
class LayerBase
{
public:
    Tensor<> *input;
    Tensor<> *output;

    Tensor<> *dInput;
    Tensor<> *dOutput;

    virtual void forward() = 0;
    virtual void backward() = 0;

    virtual void updateParams(const TrainerBase &trainer) = 0;

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
