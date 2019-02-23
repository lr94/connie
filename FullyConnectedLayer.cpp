#include "FullyConnectedLayer.hpp"

FullyConnectedLayer::FullyConnectedLayer(unsigned numNeurons)
{
    output = new Vol<>(numNeurons, 1, 1);
    dOutput = new Vol<>(numNeurons, 1, 1);
    input = nullptr;
}

FullyConnectedLayer::~FullyConnectedLayer()
{
    weights.clear();
    delete output;
    delete dOutput;
}

void FullyConnectedLayer::forward()
{
    unsigned neurons = numNeurons();
    for (unsigned i = 0; i < neurons; i++)
    {
        float value = *input * weights[i] + biases[i];
        output->set(i, value); // Possible because width and height are 1, otherwise we should have used [i][0][0]
    }
}

void FullyConnectedLayer::backward()
{
    size_t inputSize = input->getDataSize();
    unsigned neurons = numNeurons();

    for (unsigned i = 0; i < neurons; i++)
    {
        float g_i = dOutput->get(i);

        // Compute gradient w.r.t. bias
        dBiases[i] = g_i; // dB = gradient

        Vol<> &dNeuronWeights = dWeights[i];

        // Compute gradient w.r.t. weights
        for (unsigned j = 0; j < inputSize; j++) // dW = gradient * h
            dNeuronWeights.set(j, g_i * input->get(j));
    }

    // Compute gradient w.r.t. input
    for (unsigned j = 0; j < inputSize; j++)
    {
        float sum = 0;
        for (unsigned i = 0; i < neurons; i++)
            sum += dOutput->get(i) * weights[i].get(j); //dX = W * gradient
        dInput->set(j, sum);
    }
}

void FullyConnectedLayer::updateParams(const TrainerBase &trainer)
{
    trainer.updateLayerParams(biases, dBiases);

    unsigned neurons = numNeurons();
    for (unsigned i = 0; i < neurons; i++)
        trainer.updateLayerParams(weights[i], dWeights[i]);
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
    dWeights.clear();
    biases.clear();
    dBiases.clear();

    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0, 1.0);

    // For each output unit
    unsigned neurons = numNeurons();
    for (unsigned i = 0; i < neurons; i++)
    {
        weights.emplace_back(Vol<>::random(input->depth(), input->height(), input->width()));
        dWeights.emplace_back(Vol<>(input->depth(), input->height(), input->width()));
        biases.emplace_back(distribution(generator));
    }

    dBiases.reserve(biases.size());
}