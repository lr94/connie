#ifndef CNN_POOLINGLAYERBASE_HPP
#define CNN_POOLINGLAYERBASE_HPP

#include "LayerBase.hpp"

class PoolingLayerBase : public LayerBase
{
public:
    PoolingLayerBase(unsigned windowSize, unsigned stride, unsigned padding);
    PoolingLayerBase(unsigned windowWidth, unsigned windowHeight, unsigned strideX, unsigned strideY,
                    unsigned paddingX, unsigned paddingY);

protected:
    void prepend(LayerBase *previousLayer) override;

private:
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


#endif //CNN_POOLINGLAYERBASE_HPP
