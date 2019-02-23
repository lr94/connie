#ifndef CNN_NET_HPP
#define CNN_NET_HPP

#include <vector>
#include "Vol.hpp"
#include "LayerBase.hpp"

class Net
{
public:
    Net() = default;
    ~Net() = default;

    Net &appendLayer(LayerBase *layer);
    void forward();
    void backward();

    Vol<> &getInput();
    Vol<> &getOutput();
    float getLoss();
private:
    std::vector<LayerBase*> layers;

    Vol<> *input;
    Vol<> *output;
};


#endif //CNN_NET_HPP
