#include "PoolingLayerBase.hpp"

PoolingLayerBase::PoolingLayerBase(unsigned windowSize, unsigned stride, unsigned padding)
        : PoolingLayerBase(windowSize, windowSize, stride, stride, padding, padding) {}

PoolingLayerBase::PoolingLayerBase(unsigned windowWidth, unsigned windowHeight, unsigned strideX, unsigned strideY,
        unsigned paddingX, unsigned paddingY) : windowWidth(windowWidth), windowHeight(windowHeight), strideX(strideX),
        strideY(strideY), paddingX(paddingX), paddingY(paddingY) {}