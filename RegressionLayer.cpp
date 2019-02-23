//
// Created by luca on 23/02/19.
//

#include <vector>
#include "RegressionLayer.hpp"

RegressionLayer::RegressionLayer()
{
    y = input = output = dInput = dOutput = nullptr;
}

RegressionLayer::~RegressionLayer()
{
    delete y;
}

void RegressionLayer::forward()
{
    // Nothing to do
}

void RegressionLayer::backward()
{
    loss = 0;

    size_t size = y->size();
    for (unsigned i = 0; i < size; i++)
    {
        float delta = input->get(i) - y->get(i);
        dInput->set(i, delta);
        loss += delta * delta;
    }
    loss /= 2.0f;
}

Tensor<> &RegressionLayer::target()
{
    return *(this->y);
}

float RegressionLayer::getLoss()
{
    return loss;
}

void RegressionLayer::prepend(LayerBase *previousLayer)
{
    LayerBase::prepend(previousLayer);

    output = input;

    delete y;
    y = new Tensor<>(output->depth(), output->height(), output->width());
}