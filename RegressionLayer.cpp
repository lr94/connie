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

    size_t size = y->getDataSize();
    for (unsigned i = 0; i < size; i++)
    {
        float delta = input->get(i) - y->get(i);
        dInput->set(i, delta);
        loss += delta * delta;
    }
    loss /= 2.0f;
}

void RegressionLayer::setY(const std::vector<float> &y)
{
    size_t size = y.size();

    for (unsigned i = 0; i < size; i++)
        this->y->set(i, y[i]);
}

void RegressionLayer::setY(const Vol<> &y)
{
    size_t size = y.getDataSize();

    for (unsigned i = 0; i < size; i++)
        this->y->set(i, y.get(i));
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
    y = new Vol<>(output->depth(), output->height(), output->width());
}