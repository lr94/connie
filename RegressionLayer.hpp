#ifndef CNN_REGRESSIONLAYER_HPP
#define CNN_REGRESSIONLAYER_HPP

#include <vector>
#include "Tensor.hpp"
#include "LossLayerBase.hpp"

class RegressionLayer : public LossLayerBase
{
public:
    RegressionLayer();
    ~RegressionLayer();

    void forward() override;
    void backward() override;

    void updateParams(const TrainerBase &trainer) override {}

    Tensor<> &target();

    float getLoss() override;

protected:
    void prepend(LayerBase *previousLayer) override;
    Tensor<> *y;
    float loss;
};

#endif //CNN_REGRESSIONLAYER_HPP
