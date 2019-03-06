#ifndef CONNIE_LOSSLAYERBASE_HPP
#define CONNIE_LOSSLAYERBASE_HPP

#include "LayerBase.hpp"

class LossLayerBase : public LayerBase
{
public:
    virtual float getLoss() = 0;
};

#endif //CONNIE_LOSSLAYERBASE_HPP
