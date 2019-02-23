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
    Vol<> v(2, 1, 1);
    v = {0.7, -9.6};

    InputLayer inputLayer(2, 1, 1);
    FullyConnectedLayer fcc1(4);
    FullyConnectedLayer fcc2(8);
    FullyConnectedLayer fcc3(3);
    SigmoidLayer s;
    RegressionLayer r;

    Net net;
    net.appendLayer(&inputLayer)
       .appendLayer(&fcc1)
       .appendLayer(&fcc2)
       .appendLayer(&fcc3)
       .appendLayer(&s)
       .appendLayer(&r);

    Vol<> &input = net.getInput();
    input[0][0][0] = 0.7;
    input[1][0][0] = -9.6;

    r.setY(std::vector<float>{0, 0, 0});

    net.forward();
    net.backward();

    std::cout << net.getOutput() << std::endl;

    std::cout << "Loss: " << net.getLoss() << std::endl;

    return 0;
}