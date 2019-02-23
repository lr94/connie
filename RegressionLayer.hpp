#ifndef CNN_REGRESSIONLAYER_HPP
#define CNN_REGRESSIONLAYER_HPP

#include <vector>
#include "Vol.hpp"
#include "LossLayerBase.hpp"

class RegressionLayer : public LossLayerBase
{
public:
    RegressionLayer();
    ~RegressionLayer();

    void forward() override;
    void backward() override;

    void setY(const std::vector<float> &y);
    void setY(const Vol<> &y);
    float getLoss() override;

protected:
    void prepend(LayerBase *previousLayer) override;
    Vol<> *y;
    float loss;
};

#endif //CNN_REGRESSIONLAYER_HPP
