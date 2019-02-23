#ifndef CNN_SIGMOIDLAYER_HPP
#define CNN_SIGMOIDLAYER_HPP

#include <cmath>
#include "Tensor.hpp"
#include "LayerBase.hpp"

class SigmoidLayer : public LayerBase
{
public:
    SigmoidLayer()
    {
        input = output = dInput = dOutput = nullptr;
    }

    ~SigmoidLayer()
    {
        delete output;
        delete dOutput;
    }

    void forward() override
    {
        size_t inputSize = input->getDataSize();

        for (unsigned i = 0; i < inputSize; i++)
            output->set(i, sigmoid(input->get(i)));
    }

    void backward() override
    {
        size_t inputSize = input->getDataSize();

        for (unsigned i = 0; i < inputSize; i++)
            dInput->set(i, dOutput->get(i) * (1.0f - output->get(i)) * output->get(i));
    }

    void updateParams(const TrainerBase &trainer) override {}

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

private:
    static inline float sigmoid(float x)
    {
        return 1.0f / (1.0f + std::exp(-x));
    }
};


#endif //CNN_SIGMOIDLAYER_HPP
