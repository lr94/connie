#ifndef CNN_TANHLAYER_HPP
#define CNN_TANHLAYER_HPP

#include <cmath>
#include "LayerBase.hpp"

class TanhLayer : public LayerBase
{
public:
    TanhLayer()
    {
        input = output = dInput = dOutput = nullptr;
    }

    ~TanhLayer()
    {
        delete output;
        delete dOutput;
    }

    void forward() override
    {
        size_t inputSize = input->getDataSize();

        for (unsigned i = 0; i < inputSize; i++)
            output->set(i, std::tanh(input->get(i)));
    }

    void backward() override
    {
        size_t inputSize = input->getDataSize();

        for (unsigned i = 0; i < inputSize; i++)
        {
            float v = output->get(i);
            dInput->set(i, dOutput->get(i) * (1.0f - v * v));
        }
    }

protected:
    void prepend(LayerBase *previousLayer) override
    {
        LayerBase::prepend(previousLayer);

        delete output;
        delete dOutput;

        // The output has the same shape of the input
        output = new Tensor<>(input->depth(), input->height(), input->width());
        dOutput = new Tensor<>(input->depth(), input->height(), input->width());
    }
};


#endif //CNN_TANHLAYER_HPP
