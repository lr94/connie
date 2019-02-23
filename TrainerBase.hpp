#ifndef CNN_TRAINERBASE_HPP
#define CNN_TRAINERBASE_HPP

#include <vector>
#include "Net.hpp"

class TrainerBase
{
    virtual void changeParams(std::vector<float> &params, std::vector<float> &gradient) = 0;
    virtual void changeParams(Vol<> &params, Vol<> &gradient) = 0;
};


#endif //CNN_TRAINERBASE_HPP
