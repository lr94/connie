#include <iostream>
#include "Vol.hpp"

#include "Net.hpp"
#include "InputLayer.hpp"
#include "FullyConnectedLayer.hpp"
#include "SigmoidLayer.hpp"
#include "SoftmaxLayer.hpp"
#include "RegressionLayer.hpp"

int main()
{
    InputLayer inputLayer(2, 1, 1);
    FullyConnectedLayer fcc1(4);
    SigmoidLayer sig1;
    FullyConnectedLayer fcc2(1);
    RegressionLayer r;

    Net network;
    network.appendLayer(inputLayer)
           .appendLayer(fcc1)
           .appendLayer(sig1)
           .appendLayer(fcc2)
           .appendLayer(r);

    r.setY(std::vector<float>{0});
    network.forward();
    network.backward();

    std::cout << network.getOutput() << std::endl;
    std::cout << "Loss: " << network.getLoss() << std::endl;

    return 0;
}