#ifndef CNN_TRAINERBASE_HPP
#define CNN_TRAINERBASE_HPP

#include <vector>
#include "Net.hpp"

class TrainerBase
{
public:
    TrainerBase(Net &network) : net(network), networkLayers(network.layers) {}

    virtual void train() = 0;

private:
    Net &net;
    std::vector<LayerBase*> &networkLayers;
};


#endif //CNN_TRAINERBASE_HPP
