#ifndef CNN_CONVOLUTIONALLAYER_HPP
#define CNN_CONVOLUTIONALLAYER_HPP

#include <vector>
#include "Tensor.hpp"
#include "LayerBase.hpp"
#include "TrainerBase.hpp"

class ConvolutionalLayer : public LayerBase
{
public:
    ConvolutionalLayer(unsigned kernels, unsigned kernelSize, unsigned stride, unsigned padding);
    ConvolutionalLayer(unsigned kernels, unsigned kernelWidth, unsigned kernelHeight, unsigned strideX, unsigned strideY,
                       unsigned paddingX, unsigned paddingY);

    ~ConvolutionalLayer();

    void forward() override;

    void backward() override;

    void updateParams(const TrainerBase &trainer) override;

    bool save(std::ostream &stream) override;
    bool load(std::istream &stream) override;

protected:
    void prepend(LayerBase *previousLayer) override;

private:
    void initAdditionalMemory(unsigned additionalMemory) override;

    unsigned kernelCount;
    unsigned kernelWidth;
    unsigned kernelHeight;
    unsigned strideX;
    unsigned strideY;
    unsigned padX;
    unsigned padY;

    unsigned inputHeight = 0;
    unsigned inputWidth = 0;
    unsigned inputDepth = 0;

    unsigned outputHeight = 0;
    unsigned outputWidth = 0;
    unsigned outputDepth = 0;

    std::vector<Tensor<>> kernels;
    std::vector<float> biases;

    std::vector<Tensor<>> dKernels;
    std::vector<float> dBiases;

    // Additional memory for the trainer, which could want to store other info for each parameter
    // For example SGD with momentum needs to store the old delta vector, so it needs one vector for the bias
    // and one vector of tensors for the kernels. For example additionalMemBiases[0] is a vector of additional values for the
    // bias 0
    // additionalMemKernels:
    //      Index order: kernelIndex, additionalMemoryIndex
    // additionalMemBiases:
    //      Index order: additionalMemoryIndex, kernelIndex
    // They look reversed but they are not: additionalMemBiases[.] is in fact a monodimensional tensor (std::vector<float>)
    std::vector<std::vector<Tensor<>>> additionalMemKernels;
    std::vector<std::vector<float>> additionalMemBiases;
};


#endif //CNN_CONVOLUTIONALLAYER_HPP
