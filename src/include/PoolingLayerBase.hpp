#ifndef CNN_MAXPOOLINGLAYER_HPP
#define CNN_MAXPOOLINGLAYER_HPP

#include "LayerBase.hpp"

class PoolingLayerBase : LayerBase
{
public:
    PoolingLayerBase(unsigned windowSize, unsigned stride, unsigned padding);
    PoolingLayerBase(unsigned windowWidth, unsigned windowHeight, unsigned strideX, unsigned strideY,
                    unsigned paddingX, unsigned paddingY);

private:
    unsigned windowWidth;
    unsigned windowHeight;
    unsigned strideX;
    unsigned strideY;
    unsigned paddingX;
    unsigned paddingY;
};


#endif //CNN_MAXPOOLINGLAYER_HPP
