#ifndef CNN_FULLYCONNECTEDLAYER_HPP
#define CNN_FULLYCONNECTEDLAYER_HPP

#include <vector>

#include "Tensor.hpp"
#include "LayerBase.hpp"

class FullyConnectedLayer : public LayerBase
{
public:
    explicit FullyConnectedLayer(unsigned numNeurons);

    ~FullyConnectedLayer();

    void forward() override;

    void backward() override;

    void updateParams(const TrainerBase &trainer) override;

    inline unsigned numNeurons() const;

protected:
    void prepend(LayerBase *previousLayer) override;

private:
    // One tensor for neuron, each tensor has the same shape of the input tensor and contains all the weights
    std::vector<Tensor<>> weights;
    std::vector<float> biases;

    std::vector<Tensor<>> dWeights;
    std::vector<float> dBiases;
};
#endif //CNN_FULLYCONNECTEDLAYER_HPP
