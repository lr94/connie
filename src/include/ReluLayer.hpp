#ifndef CNN_RELULAYER_HPP
#define CNN_RELULAYER_HPP

#include <cmath>
#include "Tensor.hpp"
#include "LayerBase.hpp"

class ReluLayer : public LayerBase
{
public:
    ReluLayer();

    ~ReluLayer();

    void forward() override;

    void backward() override;

protected:
    void prepend(LayerBase *previousLayer) override;
};


#endif //CNN_RELULAYER_HPP
