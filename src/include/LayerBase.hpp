#ifndef CNN_LAYER_HPP
#define CNN_LAYER_HPP

#include <iostream>
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

    virtual void updateParams(const TrainerBase &trainer);

    void append(LayerBase *nextLayer);

    virtual bool save(std::ostream &stream);
    virtual bool load(std::istream &stream);

protected:
    virtual void prepend(LayerBase *previousLayer);

    bool writeFloat(std::ostream &stream, float value);

    bool readFloat(std::istream &stream, float &value);
};
#endif
