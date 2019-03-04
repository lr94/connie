#include "TrainerBase.hpp"

TrainerBase::TrainerBase(Net &network, unsigned batchSize) : TrainerBase(network, batchSize, 0) {}

TrainerBase::TrainerBase(Net &network, unsigned batchSize, unsigned additionalMemory)
        : net(network), layers(network.layers), batchSize(batchSize)
{
    if (batchSize == 0)
        throw std::runtime_error("Invalid batch size");

    for (auto &layerPtr : layers)
        layerPtr->initAdditionalMemory(additionalMemory);
}

void TrainerBase::train()
{
    iteration++;

    net.forward();
    net.backward();

    lossAccumulator += net.getLoss();
    // At the end of each minibatch
    if (iteration % batchSize == 0)
    {
        // Compute average loss
        loss = lossAccumulator / batchSize;
        lossAccumulator = 0;

        // Update the parameters
        for (auto currentLayer = layers.rbegin(); currentLayer != layers.rend(); currentLayer++)
            (*currentLayer)->updateParams(*this);
    }
}

float TrainerBase::getLoss() const
{
    return loss;
}