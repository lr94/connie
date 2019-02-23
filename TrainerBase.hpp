#ifndef CNN_TRAINERBASE_HPP
#define CNN_TRAINERBASE_HPP

#include <vector>
#include "Tensor.hpp"

class TrainerBase
{
public:
    virtual void updateLayerParams(std::vector<float> &params, std::vector<float> &gradient) const = 0;
    virtual void updateLayerParams(Tensor<> &params, Tensor<> &gradient) const = 0;
};


#endif //CNN_TRAINERBASE_HPP
