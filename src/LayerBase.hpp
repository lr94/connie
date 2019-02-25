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

    virtual void updateParams(const TrainerBase &trainer) {}

    void append(LayerBase *nextLayer)
    {
        nextLayer->prepend(this);
    }

    virtual bool save(std::ostream &stream) { return true; }
    virtual bool load(std::istream &stream) { return true; }

protected:
    virtual void prepend(LayerBase *previousLayer)
    {
        input = previousLayer->output;
        dInput = previousLayer->dOutput;
    }

    bool writeFloat(std::ostream &stream, float value)
    {
        stream.write(reinterpret_cast<char *>(&value), sizeof(value));

        return stream.good();
    }

    bool readFloat(std::istream &stream, float &value)
    {
        stream.read(reinterpret_cast<char *>(&value), sizeof(value));

        return stream.good();
    }
};
#endif
