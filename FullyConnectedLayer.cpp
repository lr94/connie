#include "FullyConnectedLayer.hpp"

FullyConnectedLayer::FullyConnectedLayer(unsigned numNeurons)
{
    output = new Vol<>(numNeurons, 1, 1);
    input = nullptr;
}

FullyConnectedLayer::~FullyConnectedLayer()
{
    weights.clear();
    delete output;
}

void FullyConnectedLayer::forward()
{
    unsigned neurons = numNeurons();
    for (unsigned i = 0; i < neurons; i++)
    {
        float value = *input * weights[i] + biases[i];
        (*output)[i] = value; // Possible because width and height are 1, otherwise we should have used [i][0][0]
    }
}

void FullyConnectedLayer::backward()
{
    unsigned neurons = numNeurons();
    for (unsigned i = 0; i < neurons; i++)
    {

    }
}

inline unsigned FullyConnectedLayer::numNeurons() const
{
    return output->depth();
}

void FullyConnectedLayer::prepend(LayerBase *previousLayer)
{
    LayerBase::prepend(previousLayer);

    // Clear weights and biases
    weights.clear();
    biases.clear();

    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0, 1.0);

    // For each output unit
    unsigned neurons = numNeurons();
    for (unsigned i = 0; i < neurons; i++)
    {
        weights.emplace_back(Vol<>::random(input->depth(), input->height(), input->width()));
        biases.emplace_back(distribution(generator));
    }
}