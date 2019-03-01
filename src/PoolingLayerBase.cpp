#include "PoolingLayerBase.hpp"

PoolingLayerBase::PoolingLayerBase(unsigned windowSize, unsigned stride, unsigned padding)
        : PoolingLayerBase(windowSize, windowSize, stride, stride, padding, padding) {}

PoolingLayerBase::PoolingLayerBase(unsigned windowWidth, unsigned windowHeight, unsigned strideX, unsigned strideY,
        unsigned paddingX, unsigned paddingY) : windowWidth(windowWidth), windowHeight(windowHeight), strideX(strideX),
        strideY(strideY), padX(paddingX), padY(paddingY) {}

void PoolingLayerBase::prepend(LayerBase *previousLayer)
{
    LayerBase::prepend(previousLayer);

    inputHeight = input->height();
    inputWidth = input->width();
    inputDepth = input->depth();

    outputHeight = (inputHeight + 2 * padY - windowHeight) / strideY + 1;
    outputWidth = (inputWidth + 2 * padX - windowWidth) / strideX + 1;
    outputDepth = inputDepth;

    output = new Tensor<>(outputDepth, outputHeight, outputWidth);
    dOutput = new Tensor<>(outputDepth, outputHeight, outputWidth);
}