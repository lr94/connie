#ifndef CNN_RELULAYER_HPP
#define CNN_RELULAYER_HPP

#include <cmath>
#include "Tensor.hpp"
#include "LayerBase.hpp"

class ReluLayer : public LayerBase
{
public:
    ReluLayer()
    {
        input = output = dInput = dOutput = nullptr;
    }

    ~ReluLayer()
    {
        delete output;
        delete dOutput;
    }

    void forward() override
    {
        size_t inputSize = input->getDataSize();

        for (unsigned i = 0; i < inputSize; i++)
            output->set(i, relu(input->get(i)));
    }

    void backward() override
    {
        size_t inputSize = input->getDataSize();

        for (unsigned i = 0; i < inputSize; i++)
            dInput->set(i, dOutput->get(i) * relu_derivative(output->get(i)));
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
    static inline float relu(float x)
    {
        return std::fmaxf(0, x);
    }

    static inline float relu_derivative(float x)
    {
        if (x < 0.0f)
            return 0.0f;
        else
            return 1.0f;
    }
};


#endif //CNN_RELULAYER_HPP
