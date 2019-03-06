#ifndef CONNIE_POOLINGLAYERBASE_HPP
#define CONNIE_POOLINGLAYERBASE_HPP

#include "LayerBase.hpp"

class PoolingLayerBase : public LayerBase
{
public:
    PoolingLayerBase(unsigned windowSize, unsigned stride, unsigned padding);
    PoolingLayerBase(unsigned windowWidth, unsigned windowHeight, unsigned strideX, unsigned strideY,
                    unsigned paddingX, unsigned paddingY);

protected:
    void prepend(LayerBase *previousLayer) override;

protected:
    unsigned windowWidth;
    unsigned windowHeight;
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
};


#endif //CONNIE_POOLINGLAYERBASE_HPP
