#include "TrainerBase.hpp"

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