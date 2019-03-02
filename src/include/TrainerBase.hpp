#ifndef CNN_TRAINERBASE_HPP
#define CNN_TRAINERBASE_HPP

#include <vector>
#include "Tensor.hpp"
#include "Net.hpp"

class TrainerBase
{
public:
    explicit TrainerBase(Net &network) : net(network), layers(network.layers) {}

    /**
     * Performs a step of the optimization algorithm. The input and the expected output (target) must have already been
     * set
     */
    virtual void train();

    /**
     * Allows a layer to update its parameters using the optimizer
     *
     * @param params
     * @param gradient
     */
    virtual void updateLayerParams(std::vector<float> &params, std::vector<float> &gradient) const = 0;

    /**
     * Allows a layer to update its parameters using the optimizer
     *
     * @param params
     * @param gradient
     */
    virtual void updateLayerParams(Tensor<> &params, Tensor<> &gradient) const = 0;

protected:
    /**
     * The network associated with the trainer
     */
    Net &net;

    /**
     * The layers of the network
     */
    std::vector<std::shared_ptr<LayerBase>> &layers;

    /**
     * Batch size, if needed
     */
    unsigned batchSize = 1;

    /**
     * Step counter
     */
    unsigned long long iteration = 0;

    /**
     * Computed loss
     */
    float loss = 0.0f;

    /**
     * Loss accumulator to calculate the batch average loss
     */
    float lossAccumulator = 0.0f;
};


#endif //CNN_TRAINERBASE_HPP
