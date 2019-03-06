#ifndef CONNIE_TANHLAYER_HPP
#define CONNIE_TANHLAYER_HPP

#include "LayerBase.hpp"

class TanhLayer : public LayerBase
{
public:
    TanhLayer();

    ~TanhLayer();

    void forward() override;

    void backward() override;

protected:
    void prepend(LayerBase *previousLayer) override;
};


#endif //CONNIE_TANHLAYER_HPP
