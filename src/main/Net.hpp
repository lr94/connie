#ifndef CNN_NET_HPP
#define CNN_NET_HPP

#include <vector>
#include <memory>
#include "Tensor.hpp"
#include "LayerBase.hpp"

class Net
{
public:
    friend class TrainerBase;

    Net() = default;
    ~Net() = default;

    Net &appendLayer(std::shared_ptr<LayerBase> layer);
    void forward();
    void backward();

    Tensor<> &getInput();
    Tensor<> &getOutput();
    float getLoss();
private:
    std::vector<std::shared_ptr<LayerBase>> layers;

    Tensor<> *input;
    Tensor<> *output;
};


#endif //CNN_NET_HPP
