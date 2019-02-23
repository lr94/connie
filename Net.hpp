#ifndef CNN_NET_HPP
#define CNN_NET_HPP

#include <vector>
#include "Vol.hpp"
#include "LayerBase.hpp"
#include "TrainerBase.hpp"

class Net
{
public:
    friend class TrainerBase;

    Net() = default;
    ~Net() = default;

    Net &appendLayer(LayerBase &layer);
    void forward();
    void backward();
    void train(TrainerBase &trainer);

    Vol<> &getInput();
    Vol<> &getOutput();
    float getLoss();
private:
    std::vector<LayerBase*> layers;

    Vol<> *input;
    Vol<> *output;
};


#endif //CNN_NET_HPP
