#ifndef CNN_TRAINERBASE_HPP
#define CNN_TRAINERBASE_HPP

#include <vector>
#include "Tensor.hpp"
#include "Net.hpp"

class TrainerBase
{
public:
    explicit TrainerBase(Net &network) : net(network), layers(network.layers) {}

    virtual void train() = 0;
    virtual void updateLayerParams(std::vector<float> &params, std::vector<float> &gradient, std::vector<float> &memory) const = 0;
    virtual void updateLayerParams(Tensor<> &params, Tensor<> &gradient, Tensor<> &memory) const = 0;

protected:
    Net &net;
    std::vector<std::shared_ptr<LayerBase>> &layers;
};


#endif //CNN_TRAINERBASE_HPP
