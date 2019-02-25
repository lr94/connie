#ifndef CNN_NET_HPP
#define CNN_NET_HPP

#include <iostream>
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

    bool save(const char *filename);
    bool save(std::ostream &stream);
    bool load(const char *filename);
    bool load(std::istream &stream);
private:
    std::vector<std::shared_ptr<LayerBase>> layers;

    Tensor<> *input;
    Tensor<> *output;
};


#endif //CNN_NET_HPP
