//
// Created by luca on 28/02/19.
//

#include "ConvolutionalLayer.hpp"

ConvolutionalLayer::ConvolutionalLayer(unsigned kernels, unsigned kernelSize, unsigned stride)
    : ConvolutionalLayer(kernels, kernelSize, kernelSize, stride, stride)
{

}

ConvolutionalLayer::ConvolutionalLayer(unsigned kernels, unsigned kernelWidth, unsigned kernelHeight,
        unsigned strideX, unsigned strideY) : kernelCount(kernels),
        kernelWidth(kernelWidth), kernelHeight(kernelHeight), strideX(strideX), strideY(strideY)
{
    output = input = dOutput = dInput = nullptr;
}

ConvolutionalLayer::~ConvolutionalLayer()
{
    kernels.clear();
    delete[] output;
    delete[] dOutput;
}

void ConvolutionalLayer::forward()
{

}

void ConvolutionalLayer::backward()
{

}

void ConvolutionalLayer::updateParams(const TrainerBase &trainer)
{

}

void ConvolutionalLayer::prepend(LayerBase *previousLayer)
{
    LayerBase::prepend(previousLayer);


}

bool ConvolutionalLayer::save(std::ostream &stream)
{
    return false;
}

bool ConvolutionalLayer::load(std::istream &stream)
{
    return false;
}