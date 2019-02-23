#ifndef CNN_NET_HPP
#define CNN_NET_HPP

#include <vector>
#include "Tensor.hpp"
#include "LayerBase.hpp"
#include "TrainerBase.hpp"

class Net
{
public:
    Net() = default;
    ~Net() = default;

    Net &appendLayer(LayerBase &layer);
    void forward();
    void backward();
    void train(TrainerBase &trainer);

    Tensor<> &getInput();
    Tensor<> &getOutput();
    float getLoss();
private:
    std::vector<LayerBase*> layers;

    Tensor<> *input;
    Tensor<> *output;
};


#endif //CNN_NET_HPP
