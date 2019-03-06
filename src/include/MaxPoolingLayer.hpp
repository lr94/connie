#ifndef CONNIE_MAXPOOLINGLAYER_HPP
#define CONNIE_MAXPOOLINGLAYER_HPP

#include "PoolingLayerBase.hpp"

class MaxPoolingLayer : public PoolingLayerBase
{
public:
    MaxPoolingLayer(unsigned windowSize, unsigned stride, unsigned padding);
    MaxPoolingLayer(unsigned windowWidth, unsigned windowHeight, unsigned strideX, unsigned strideY,
                     unsigned paddingX, unsigned paddingY);
    ~MaxPoolingLayer();

    void forward() override;
    void backward() override;
};


#endif //CONNIE_MAXPOOLINGLAYER_HPP
