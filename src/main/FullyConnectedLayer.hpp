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
    // One weight tensor per neuron, each tensor has the same shape of the input tensor and contains all the weights
    // Weights and biases
    std::vector<Tensor<>> weights;
    std::vector<float> biases;

    // Weight gradient tensors and bias gradients
    std::vector<Tensor<>> dWeights;
    std::vector<float> dBiases;

    // Generic additional memory for weight and biases; it can be used as gradient accumulator for stochastic gradient
    // descent and similar training methods
    std::vector<Tensor<>> memWeights;
    std::vector<float> memBiases;
};
#endif //CNN_FULLYCONNECTEDLAYER_HPP
