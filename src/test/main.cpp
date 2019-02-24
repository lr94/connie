#include <iostream>
#include <memory>
#include "Tensor.hpp"

#include "Net.hpp"
#include "InputLayer.hpp"
#include "FullyConnectedLayer.hpp"
#include "SigmoidLayer.hpp"
#include "ReluLayer.hpp"
#include "SoftmaxLayer.hpp"
#include "RegressionLayer.hpp"

#include "SGDTrainer.hpp"

int main()
{
    // Build the network
    std::shared_ptr<RegressionLayer> r = std::make_shared<RegressionLayer>();

    Net network;
    network.appendLayer(std::make_shared<InputLayer>(2, 1, 1))
           .appendLayer(std::make_shared<FullyConnectedLayer>(4))
           .appendLayer(std::make_shared<SigmoidLayer>())
           .appendLayer(std::make_shared<FullyConnectedLayer>(1))
           .appendLayer(r);

    Tensor<> &input = network.getInput();
    Tensor<> &output = network.getOutput();
    Tensor<> &target = r->target();

    // Initialize the trainer
    SGDTrainer trainer(network, 0.001f);

    // Training data
    float x[][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    float y[] = {0, 1, 1, 0};
    int n = 4;

    for (unsigned i = 0; i < 500000; i++)
    {
        input.set(0, x[i % n][0]);
        input.set(1, x[i % n][1]);
        target.set(0, y[i % n]);

        trainer.train();

        if (i % 1000 == 0)
            std::cout << i << ": " << network.getLoss() << std::endl;
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