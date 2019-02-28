//
// Created by luca on 28/02/19.
//

#ifndef CNN_CONVOLUTIONALLAYER_HPP
#define CNN_CONVOLUTIONALLAYER_HPP

#include <vector>
#include "Tensor.hpp"
#include "LayerBase.hpp"
#include "TrainerBase.hpp"

class ConvolutionalLayer : public LayerBase
{
public:
    ConvolutionalLayer(unsigned kernels, unsigned kernelSize, unsigned stride);
    ConvolutionalLayer(unsigned kernels, unsigned kernelWidth, unsigned kernelHeight, unsigned strideX, unsigned strideY);

    ~ConvolutionalLayer();

    void forward() override;

    void backward() override;

    void updateParams(const TrainerBase &trainer) override;

    inline unsigned numNeurons() const;

    bool save(std::ostream &stream) override;
    bool load(std::istream &stream) override;

protected:
    void prepend(LayerBase *previousLayer) override;

private:
    unsigned kernelCount;
    unsigned kernelWidth;
    unsigned kernelHeight;
    unsigned inputWidth;
    unsigned inputHeight;
    unsigned strideX;
    unsigned strideY;

    unsigned outputWidth;
    unsigned outputHeight;
    std::vector<Tensor<>> kernels;
};


#endif //CNN_CONVOLUTIONALLAYER_HPP
