#include "TrainerBase.hpp"

TrainerBase::TrainerBase(Net &network, unsigned additionalMemory) : net(network), layers(network.layers)
{
    for (auto &layerPtr : layers)
        layerPtr->initAdditionalMemory(additionalMemory);
}

void TrainerBase::train()
{
    iteration++;

    net.forward();
    net.backward();

    lossAccumulator += net.getLoss();
    if (iteration % batchSize == 0)
    {
        loss = lossAccumulator / batchSize;
        lossAccumulator = 0;
    }

    for (auto currentLayer = layers.rbegin(); currentLayer != layers.rend(); currentLayer++)
        (*currentLayer)->updateParams(*this);
}