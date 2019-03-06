#ifndef CONNIE_FULLYCONNECTEDLAYER_HPP
#define CONNIE_FULLYCONNECTEDLAYER_HPP

#include <ios>
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

    bool save(std::ostream &stream) override;
    bool load(std::istream &stream) override;

protected:
    void prepend(LayerBase *previousLayer) override;

private:
    void initAdditionalMemory(unsigned additionalMemory) override;

    // One weight tensor per neuron, each tensor has the same shape of the input tensor and contains all the weights
    // Weights and biases
    std::vector<Tensor<>> weights;
    std::vector<float> biases;

    // Weight gradient tensors and bias gradients
    std::vector<Tensor<>> dWeights;
    std::vector<float> dBiases;

    // Additional memory for the trainer, which could want to store other info for each parameter
    // For example SGD with momentum needs to store the old delta vector, so it needs one vector for the bias
    // and one vector of tensors for the weights. For example additionalMemBiases[0] is a vector of additional values for the
    // bias 0
    // additionalMemWeights:
    //      Index order: weightIndex, additionalMemoryIndex
    // additionalMemBiases:
    //      Index order: additionalMemoryIndex, unitIndex
    // They look reversed but they are not: additionalMemBiases[.] is in fact a monodimensional tensor (std::vector<float>)
    std::vector<std::vector<Tensor<>>> additionalMemWeights;
    std::vector<std::vector<float>> additionalMemBiases;
};
#endif //CONNIE_FULLYCONNECTEDLAYER_HPP
