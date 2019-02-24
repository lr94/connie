#ifndef CNN_TRAINERBASE_HPP
#define CNN_TRAINERBASE_HPP

#include <vector>
#include "Tensor.hpp"

class TrainerBase
{
public:
    virtual void updateLayerParams(std::vector<float> &params, std::vector<float> &gradient, std::vector<float> &memory) const = 0;
    virtual void updateLayerParams(Tensor<> &params, Tensor<> &gradient, Tensor<> &memory) const = 0;
};


#endif //CNN_TRAINERBASE_HPP
