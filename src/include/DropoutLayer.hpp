#ifndef CONNIE_DROPOUTLAYER_HPP
#define CONNIE_DROPOUTLAYER_HPP

#include <random>
#include "LayerBase.hpp"

class DropoutLayer : LayerBase
{
public:
    DropoutLayer();
    DropoutLayer(float p);

    ~DropoutLayer();

    void forward() override;
    void backward() override;

protected:
    void prepend(LayerBase *previousLayer) override;

private:
    float p;

    std::default_random_engine randomEngine;
    std::bernoulli_distribution bernoulli;

    Tensor<bool> *mask;
};


#endif //CONNIE_DROPOUTLAYER_HPP
