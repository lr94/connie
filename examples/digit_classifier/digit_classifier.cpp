#include <iostream>
#include <iomanip>
#include <string>
#include <memory>

#include <gd.h>

#include "Tensor.hpp"
#include "Net.hpp"
#include "ConvolutionalLayer.hpp"
#include "MaxPoolingLayer.hpp"
#include "FullyConnectedLayer.hpp"
#include "InputLayer.hpp"
#include "ReluLayer.hpp"
#include "SoftmaxLayer.hpp"

std::string readStringArg(const std::string &argumentName, const std::string &defaultValue, int argc, char *argv[]);
Net initNetwork(std::shared_ptr<SoftmaxLayer> &softmaxLayer, int argc, char *argv[]);
bool loadImage(Tensor<> &imageTensor, std::string &filename);

const unsigned size = 28;

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cout << "Usage:" << std::endl;
        std::cout << "\t" << argv[0] << " IMAGE_FILE [--network-file NETWORK_FILE] [--network-type lenet|fc]" << std::endl;
        return EXIT_SUCCESS;
    }

    std::string networkFile = readStringArg("network-file", "network.bin", argc, argv);
    std::string imageFile = argv[1];

    std::shared_ptr<SoftmaxLayer> softmax = std::make_shared<SoftmaxLayer>();
    Net network = initNetwork(softmax, argc, argv);
    network.load(networkFile.c_str());

    Tensor<> &inputTensor = network.getInput();
    Tensor<> &outputTensor = network.getOutput();

    if (!loadImage(inputTensor, imageFile))
    {
        std::cout << "Could not load file " << imageFile << std::endl;
        return EXIT_FAILURE;
    }

    network.forward();

    std::cout << softmax->getPredictedClass() << std::endl << std::endl;

    for (unsigned i = 0; i < softmax->getNumClasses(); i++)
    {
        float percent = outputTensor[i] * 100.0f;
        std::cout << i << ": " << std::setprecision(4) << percent << " %" << std::endl;
    }
}

std::string readStringArg(const std::string &argumentName, const std::string &defaultValue, int argc, char *argv[])
{
    std::string fullName = "--" + argumentName;

    for (int i = 1; i < argc; i++)
    {
        int next_arg_index = i + 1;
        std::string current_arg = argv[i];

        if (current_arg == fullName && next_arg_index < argc)
            return std::string(argv[++i]);
    }

    return defaultValue;
}

Net initNetwork(std::shared_ptr<SoftmaxLayer> &softmaxLayer, int argc, char *argv[])
{
    std::string networkString = readStringArg("network-type", "lenet", argc, argv);
    Net network;

    network.appendLayer(std::make_shared<InputLayer>(1, size, size));

    if (networkString == "lenet")
    {
        network.appendLayer(std::make_shared<ConvolutionalLayer>(6, 5, 1, 2))
                .appendLayer(std::make_shared<ReluLayer>())
                .appendLayer(std::make_shared<MaxPoolingLayer>(2, 2, 0))
                .appendLayer(std::make_shared<ConvolutionalLayer>(16, 5, 1, 0))
                .appendLayer(std::make_shared<ReluLayer>())
                .appendLayer(std::make_shared<MaxPoolingLayer>(2, 2, 0))
                .appendLayer(std::make_shared<ConvolutionalLayer>(120, 1, 1, 0))
                .appendLayer(std::make_shared<ReluLayer>())
                .appendLayer(std::make_shared<FullyConnectedLayer>(84))
                .appendLayer(std::make_shared<ReluLayer>())
                .appendLayer(std::make_shared<FullyConnectedLayer>(10));

        std::cout << "Network: LeNet-5 convolutional network" << std::endl;
    }
    else if (networkString == "fc")
    {
        network.appendLayer(std::make_shared<FullyConnectedLayer>(size * size * 2))
                .appendLayer(std::make_shared<ReluLayer>())
                .appendLayer(std::make_shared<FullyConnectedLayer>(size * size))
                .appendLayer(std::make_shared<ReluLayer>())
                .appendLayer(std::make_shared<FullyConnectedLayer>(10));

        std::cout << "Network: Fully connected network" << std::endl;
    }
    else
    {
        std::cout << "Unknown network " << networkString << std::endl;
        std::cout << "Valid networks are: lenet, fc" << std::endl;
        exit(1);
    }

    network.appendLayer(softmaxLayer);

    return network;
}

bool loadImage(Tensor<> &imageTensor, std::string &filename)
{
    gdImagePtr img = gdImageCreateFromFile(filename.c_str());

    if (img == nullptr)
        return false;

    // auto width = static_cast<unsigned>(gdImageSX(img));
    // auto height = static_cast<unsigned>(gdImageSY(img));
    unsigned resizeWidth = size;
    unsigned resizeHeight = size;

    gdImagePtr resizedImg = gdImageScale(img, resizeWidth, resizeHeight);
    gdImageDestroy(img);

    for (unsigned x = 0; x < resizeWidth; x++)
        for (unsigned y = 0; y < resizeHeight; y++)
        {
            int color = gdImageGetPixel(resizedImg, x, y);
            float r = static_cast<float>((color >> 16) & 0xff) / 255.0f;
            float g = static_cast<float>((color >> 8) & 0xff) / 255.0f;
            float b = static_cast<float>(color & 0xff) / 255.0f;

            float val = 1.0f - (r + g + b) / 3; // We want the negative image

            imageTensor[0][y][x] = val;
        }

    gdImageDestroy(resizedImg);

    return true;
}