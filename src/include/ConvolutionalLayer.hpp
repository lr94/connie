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
};


#endif //CNN_CONVOLUTIONALLAYER_HPP
