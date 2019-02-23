#include <iostream>
#include "../main/Tensor.hpp"

#include "../main/Net.hpp"
#include "../main/InputLayer.hpp"
#include "../main/FullyConnectedLayer.hpp"
#include "../main/SigmoidLayer.hpp"
#include "../main/ReluLayer.hpp"
#include "../main/SoftmaxLayer.hpp"
#include "../main/RegressionLayer.hpp"

#include "../main/SGDTrainer.hpp"

int main()
{
    // Build the network
    InputLayer inputLayer(2, 1, 1);
    FullyConnectedLayer fcc1(4);
    SigmoidLayer activationFunction;
    FullyConnectedLayer fcc2(1);
    RegressionLayer r;

    Net network;
    network.appendLayer(inputLayer)
           .appendLayer(fcc1)
           .appendLayer(activationFunction)
           .appendLayer(fcc2)
           .appendLayer(r);

    Tensor<> &input = network.getInput();
    Tensor<> &output = network.getOutput();
    Tensor<> &target = r.target();

    // Initialize the trainer
    SGDTrainer trainer(0.001f);

    // Training data
    float x[][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    float y[] = {0, 1, 1, 0};
    int n = 4;

    for (unsigned i = 0; i < 500000; i++)
    {
        input.set(0, x[i % n][0]);
        input.set(1, x[i % n][1]);
        target.set(0, y[i % n]);

        network.train(trainer);

        if (i % 1000 == 0)
            std::cout << network.getLoss() << std::endl;
    }

    for (unsigned i = 0; i < n; i++)
    {
        input.set(0, x[i][0]);
        input.set(1, x[i][1]);
        network.forward();
        std::cout << "{" << x[i][0] << ", " << x[i][1] << "} -> " << output.get(0) << std::endl;
    }

    return 0;
}