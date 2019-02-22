#ifndef CNN_INPUTLAYER_HPP
#define CNN_INPUTLAYER_HPP

#include "Vol.hpp"
#include "LayerBase.hpp"

class InputLayer : public LayerBase
{
public:
    InputLayer(unsigned depth, unsigned height, unsigned width)
    {
        // In the input layer the input tensor and the output tensor are the same
        input = new Vol<>(depth, height, width);
        dInput = new Vol<>(depth, height, width);
        output = input;
        dOutput = dInput;
    }

    ~InputLayer()
    {
        delete input;
    }

    void forward() override {}
    void backward() override {}

protected:
    void prepend(LayerBase *previousLayer) override
    {
        // LayerBase::prepend(previousLayer);
        throw std::runtime_error("Cannot append an InputLayer");
    }
};

#endif //CNN_INPUTLAYER_HPP
