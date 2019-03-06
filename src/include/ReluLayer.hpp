#ifndef CONNIE_RELULAYER_HPP
#define CONNIE_RELULAYER_HPP

#include "LayerBase.hpp"

class ReluLayer : public LayerBase
{
public:
    ReluLayer();
    ReluLayer(float leak);

    ~ReluLayer();

    void forward() override;

    void backward() override;

protected:
    void prepend(LayerBase *previousLayer) override;

private:
    float leak;
};


#endif //CONNIE_RELULAYER_HPP
