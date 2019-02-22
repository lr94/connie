#ifndef CNN_FULLYCONNECTEDLAYER_HPP
#define CNN_FULLYCONNECTEDLAYER_HPP

#include <vector>

#include "Vol.hpp"
#include "LayerBase.hpp"

class FullyConnectedLayer : public LayerBase
{
public:
    FullyConnectedLayer(unsigned numNeurons);

    ~FullyConnectedLayer();

    void forward() override;

    void backward() override;

    inline unsigned numNeurons() const;

protected:
    void prepend(LayerBase *previousLayer) override;

private:
    // One tensor for neuron, each tensor has the same shape of the input tensor and contains all the weights
    std::vector<Vol<>> weights;
    std::vector<float> biases;

    std::vector<Vol<>> dWeights;
    std::vector<float> dBiases;
};
#endif //CNN_FULLYCONNECTEDLAYER_HPP
