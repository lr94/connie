#ifndef CNN_TRAINERBASE_HPP
#define CNN_TRAINERBASE_HPP

#include <vector>
#include "Vol.hpp"

class TrainerBase
{
    virtual void changeParams(std::vector<float> &params, std::vector<float> &gradient) const = 0;
    virtual void changeParams(Vol<> &params, Vol<> &gradient) const = 0;
};


#endif //CNN_TRAINERBASE_HPP
