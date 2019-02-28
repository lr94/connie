#ifndef CNN_LOSSLAYERBASE_HPP
#define CNN_LOSSLAYERBASE_HPP

#include "LayerBase.hpp"

class LossLayerBase : public LayerBase
{
public:
    virtual float getLoss() = 0;
};

#endif //CNN_LOSSLAYERBASE_HPP
