#ifndef CNN_TANHLAYER_HPP
#define CNN_TANHLAYER_HPP

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


#endif //CNN_TANHLAYER_HPP
