#include <catch2/catch.hpp>
#include "SigmoidLayer.hpp"
#include "TanhLayer.hpp"
#include "ReluLayer.hpp"
#include "RegressionLayer.hpp"
#include "SGDTrainer.hpp"
#include "SoftmaxLayer.hpp"
#include "FullyConnectedLayer.hpp"
#include "Net.hpp"
#include "InputLayer.hpp"
#include "MemoryStream.hpp"

TEST_CASE("Learn XOR with regression", "[learn][regression][xor][sigmoid][tanh][relu]")
{
    Net network;
    network.appendLayer(std::make_shared<InputLayer>(2, 1, 1))
           .appendLayer(std::make_shared<FullyConnectedLayer>(4))
           .appendLayer(std::make_shared<ReluLayer>());

    unsigned char *initialParams = nullptr;
    size_t len = 0;

    // Since starting with random weights the network could not converge, let's start with known ones
    SECTION("Sigmoid")
    {
        unsigned char initial[] = {
                0x8b, 0x93, 0x0e, 0x3f, 0xc7, 0x4f, 0xe3, 0xbe, 0x52, 0x93, 0xdd, 0x3f,
                0x16, 0x0d, 0x98, 0xbe, 0x7a, 0xb1, 0x5f, 0x3f, 0xbe, 0x50, 0xd9, 0x3f,
                0xaf, 0x1e, 0x88, 0x3f, 0x3c, 0x5a, 0x8e, 0x3f, 0x52, 0xff, 0x75, 0x3f,
                0xa4, 0x6d, 0xf6, 0x3f, 0xa4, 0xd9, 0x8c, 0xbf, 0x1a, 0x89, 0x0b, 0x40,
                0xeb, 0x9d, 0x7b, 0xbd, 0x9a, 0xf2, 0xc6, 0xbf, 0xf1, 0x61, 0x05, 0xbd,
                0x42, 0xd9, 0x34, 0x3e, 0x65, 0x05, 0x3a, 0x3f
        };
        initialParams = initial;
        len = sizeof(initial);
        network.appendLayer(std::make_shared<SigmoidLayer>());
    }

    SECTION("Tanh")
    {
        unsigned char initial[] = {
                0x4d, 0x3a, 0x8a, 0xbf, 0x09, 0xa1, 0x28, 0xbf, 0x27, 0xd7, 0x98, 0xbd,
                0x04, 0xf3, 0x6f, 0xbf, 0x21, 0x38, 0x39, 0x3f, 0x53, 0x10, 0x94, 0xbf,
                0x00, 0xc6, 0xd6, 0xbf, 0xd9, 0x90, 0x73, 0xbf, 0x05, 0x81, 0xf4, 0x3e,
                0x62, 0xd0, 0x9f, 0x3f, 0x73, 0x41, 0x88, 0x3f, 0x2e, 0x1b, 0x93, 0x3f,
                0x30, 0x95, 0xa9, 0x3e, 0x7a, 0x9f, 0x19, 0x3f, 0xec, 0xc6, 0xdf, 0x3c,
                0x8c, 0xb5, 0x3b, 0x3e, 0x23, 0x87, 0x39, 0x3f
        };
        initialParams = initial;
        len = sizeof(initial);
        network.appendLayer(std::make_shared<TanhLayer>());
    }

    SECTION("ReLU")
    {
        unsigned char initial[] = {
                0x83, 0x06, 0x81, 0x3e, 0xa4, 0x3e, 0x82, 0x3e, 0xda, 0x30, 0x0f, 0xbf,
                0x1b, 0xa5, 0x1a, 0x3f, 0x1e, 0x98, 0xb0, 0xbe, 0x75, 0x68, 0x4c, 0x3f,
                0x05, 0x33, 0x81, 0xbf, 0xbe, 0x37, 0x87, 0xbe, 0xb8, 0x9f, 0x38, 0xbb,
                0xcc, 0x5c, 0x25, 0x3f, 0x83, 0x07, 0x33, 0xc0, 0x47, 0xf2, 0x8f, 0xbf,
                0xe0, 0x81, 0x04, 0x3f, 0xa8, 0x7c, 0x06, 0xbf, 0x12, 0x7b, 0x6a, 0xbe,
                0xfe, 0x2d, 0xc3, 0xbe, 0x12, 0x91, 0xab, 0x3f
        };
        initialParams = initial;
        len = sizeof(initial);
        network.appendLayer(std::make_shared<ReluLayer>());
    }

    std::shared_ptr<RegressionLayer> r = std::make_shared<RegressionLayer>();
    network.appendLayer(std::make_shared<FullyConnectedLayer>(1))
           .appendLayer(r);

    MemoryStream memoryStream(initialParams, len);
    network.load(memoryStream);

    Tensor<> &input = network.getInput();
    Tensor<> &output = network.getOutput();
    Tensor<> &target = r->target();

    // Initialize the trainer
    SGDTrainer trainer(network, 0.05f, 4);

    // Training data
    float x[][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    float y[] = {0, 1, 1, 0};
    unsigned n = 4;

    for (unsigned i = 0; i < 50000; i++)
    {
        input.set(0, x[i % n][0]);
        input.set(1, x[i % n][1]);
        target.set(0, y[i % n]);

        trainer.train();
    }

    for (unsigned i = 0; i < n; i++)
    {
        input.set(0, x[i][0]);
        input.set(1, x[i][1]);
        network.forward();
        REQUIRE(y[i] == Approx(output.get(0)).margin(0.01));
    }
}

TEST_CASE("Learn XOR with classification", "[learn][classification][xor][sigmoid][tanh][relu][softmax]")
{
    Net network;
    network.appendLayer(std::make_shared<InputLayer>(2, 1, 1))
            .appendLayer(std::make_shared<FullyConnectedLayer>(4))
            .appendLayer(std::make_shared<ReluLayer>());

    unsigned char *initialParams = nullptr;
    size_t len = 0;

    // Since starting with random weights the network could not converge, let's start with known ones
    SECTION("Sigmoid")
    {
        unsigned char initial[] = {
                0x90, 0x22, 0xe8, 0x3e, 0xe2, 0x5d, 0xd2, 0xbe, 0x16, 0x2c, 0x00, 0x3d,
                0x68, 0xd0, 0xb2, 0xbd, 0xda, 0xbd, 0x28, 0xbd, 0xf6, 0xcc, 0xb4, 0xbf,
                0x33, 0x24, 0xa7, 0xbe, 0xd0, 0x52, 0x10, 0xbd, 0x38, 0x9d, 0x67, 0x3f,
                0x66, 0x6f, 0x47, 0xbe, 0x1a, 0xa9, 0x01, 0x3f, 0xfa, 0x40, 0x77, 0xbe,
                0xab, 0xf6, 0xc6, 0x3d, 0x9d, 0xff, 0x5d, 0xbf, 0xdf, 0x4e, 0x62, 0x3f,
                0xed, 0x00, 0x37, 0xbf, 0xc2, 0x74, 0xb6, 0xbd, 0xbd, 0xe4, 0xa1, 0xbe,
                0xfd, 0xa5, 0x80, 0xbd, 0x11, 0x0e, 0x8a, 0x3f, 0x43, 0x18, 0x88, 0x3f,
                0x42, 0xc6, 0x79, 0x3d
        };
        initialParams = initial;
        len = sizeof(initial);
        network.appendLayer(std::make_shared<SigmoidLayer>());
    }

    SECTION("Tanh")
    {
        unsigned char initial[] = {
                0x07, 0x3d, 0x01, 0xbf, 0x67, 0x56, 0xda, 0xbf, 0x11, 0x9a, 0xf5, 0xbe,
                0xc6, 0x7b, 0x11, 0xbf, 0xa0, 0xaa, 0x40, 0x3f, 0xe7, 0x26, 0xcd, 0xbe,
                0x0d, 0x66, 0xf2, 0xbe, 0xda, 0x9a, 0x3f, 0xbe, 0x9f, 0xe7, 0x14, 0xc0,
                0x9e, 0xe5, 0xaa, 0x3e, 0x38, 0x4f, 0x35, 0xbf, 0x7e, 0xa3, 0x05, 0x3f,
                0x9a, 0x17, 0xc6, 0xbc, 0x55, 0xa7, 0x01, 0xbd, 0x4c, 0x9d, 0x79, 0x3e,
                0xb6, 0xa0, 0xb6, 0xbe, 0x45, 0x71, 0x90, 0xbe, 0x1d, 0x79, 0x79, 0x3f,
                0x20, 0x41, 0xf6, 0x3e, 0x18, 0x77, 0x8b, 0x3f, 0x92, 0xb2, 0x31, 0x3d,
                0xa8, 0x67, 0xcb, 0x3f
        };
        initialParams = initial;
        len = sizeof(initial);
        network.appendLayer(std::make_shared<TanhLayer>());
    }

    SECTION("ReLU")
    {
        unsigned char initial[] = {
                0xa2, 0x35, 0x83, 0xbf, 0x14, 0xf4, 0x06, 0x3f, 0x2b, 0x0b, 0x97, 0x3d,
                0xbd, 0x13, 0xa1, 0xbc, 0xe2, 0x88, 0x12, 0xbf, 0x67, 0x79, 0xd4, 0xbd,
                0x4c, 0x4e, 0xea, 0xbe, 0x28, 0x5e, 0x4c, 0xbf, 0xfd, 0xf8, 0x16, 0x3f,
                0x7c, 0x3f, 0x86, 0x3e, 0x07, 0x38, 0x46, 0x3d, 0x44, 0x28, 0x9d, 0x3f,
                0x73, 0xa3, 0xfd, 0x3d, 0x8f, 0x5c, 0x8e, 0x3e, 0x57, 0x6d, 0x44, 0x3e,
                0x72, 0x50, 0x8b, 0xbe, 0xc7, 0x86, 0x74, 0xbf, 0xd4, 0xdd, 0x2f, 0xbf,
                0x57, 0x11, 0x10, 0xbf, 0x0a, 0x80, 0xb5, 0x3f, 0xb3, 0xec, 0xf1, 0x3f,
                0x0e, 0x69, 0x9e, 0x3f
        };
        initialParams = initial;
        len = sizeof(initial);
        network.appendLayer(std::make_shared<ReluLayer>());
    }

    std::shared_ptr<SoftmaxLayer> softmax = std::make_shared<SoftmaxLayer>();
    network.appendLayer(std::make_shared<FullyConnectedLayer>(2))
            .appendLayer(softmax);

    MemoryStream memoryStream(initialParams, len);
    network.load(memoryStream);

    Tensor<> &input = network.getInput();

    // Initialize the trainer
    SGDTrainer trainer(network, 0.05f, 4);

    // Training data
    float x[][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    unsigned y[] = {0, 1, 1, 0};
    unsigned n = 4;

    for (unsigned i = 0; i < 50000; i++)
    {
        input.set(0, x[i % n][0]);
        input.set(1, x[i % n][1]);
        softmax->setTargetClass(y[i % n]);

        trainer.train();
    }

    for (unsigned i = 0; i < n; i++)
    {
        input.set(0, x[i][0]);
        input.set(1, x[i][1]);
        network.forward();
        REQUIRE(y[i] == softmax->getPredictedClass());
    }
}