#include <iostream>
#include <fstream>
#include <filesystem>
#include <memory>
#include <chrono>

#include "Tensor.hpp"
#include "Net.hpp"
#include "SGDTrainer.hpp"
#include "InputLayer.hpp"
#include "FullyConnectedLayer.hpp"
#include "TanhLayer.hpp"
#include "SigmoidLayer.hpp"
#include "SoftmaxLayer.hpp"

#include "Dataset.hpp"

static void loadSample(Tensor<> &tensor, const Sample &sample);

static const unsigned size = 28;
static const char defaultDataFile[] = "../datasets/mnist/train-images-idx3-ubyte";
static const char defaultLabelsFile[] = "../datasets/mnist/train-labels-idx1-ubyte";
static const char defaultNetworkFile[] = "network.bin";

int main(int argc, char *argv[])
{
    const char *dataFile = defaultDataFile;
    const char *labelsFile = defaultLabelsFile;
    const char *networkFile = defaultNetworkFile;
    const char *logFile = nullptr;

    if (argc < 2)
        std::cout << "Usage:\n\t" << argv[0] << " [-n NETWORK_FILE] [-tl TRAINING_LABELS_FILE] [-td TRAINING_DATA_FILE] [-l LOG_FILE]" << std::endl << std::endl;
    for (unsigned i = 1; i < argc; i++)
    {
        int next_arg_index = i + 1;
        std::string current_arg = argv[i];

        if (current_arg == "-n" && next_arg_index < argc)
            networkFile = argv[++i];
        if (current_arg == "-tl" && next_arg_index < argc)
            labelsFile = argv[++i];
        if (current_arg == "-td" && next_arg_index < argc)
            dataFile = argv[++i];
        if (current_arg == "-l" && next_arg_index < argc)
            logFile = argv[++i];
    }

    std::cout << "Loading dataset from " << labelsFile << " and " << dataFile << std::endl;
    Dataset dataset("../datasets/mnist/train-images-idx3-ubyte", "../datasets/mnist/train-labels-idx1-ubyte");
    std::cout << "Loaded " << dataset.size() << " samples." << std::endl;

    // Build the network
    Net network;
    std::shared_ptr<SoftmaxLayer> softmax = std::make_shared<SoftmaxLayer>();
    network.appendLayer(std::make_shared<InputLayer>(1, size, size))
            .appendLayer(std::make_shared<FullyConnectedLayer>(size * size * 2))
            .appendLayer(std::make_shared<SigmoidLayer>())
            .appendLayer(std::make_shared<FullyConnectedLayer>(10))
            .appendLayer(softmax);
    // Load the network if necessary
    if (std::filesystem::exists(networkFile))
    {
        network.load(networkFile);
        std::cout << "Loaded network from " << networkFile << std::endl;
    }
    Tensor<> &input = network.getInput();
    Tensor<> &output = network.getOutput();

    // Init the trainer
    SGDTrainer trainer(network, 0.05f, 8);

    // Init log file if necessary
    std::ofstream log;
    if (logFile != nullptr)
    {
        log.open(logFile);
        log << "millisecondss,epoch,iteration,loss" << std::endl;
    }

    std::chrono::time_point start = std::chrono::system_clock::now();
    unsigned long long iteration = 0;
    for (unsigned epoch = 0; epoch < 1000000; epoch++)
    {
        for (auto &sample : dataset)
        {
            loadSample(input, sample);
            softmax->setTargetClass(sample.label());

            trainer.train();

            iteration++;
            if (iteration % 10 == 0)
            {
                float loss = trainer.getLoss();
                std::cout << "Epoch: " << epoch << " iteration: " << iteration << " loss: " << loss << std::endl;
                if (logFile != nullptr)
                {
                    long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start).count();
                    log << ms << "," << epoch << "," << iteration << "," << loss << std::endl;
                }
            }
            if (iteration % 500 == 0)
            {
                network.save(networkFile);
                std::cout << "Network saved in " << networkFile << std::endl;
            }
        }
    }
}

static void loadSample(Tensor<> &tensor, const Sample &sample)
{
    unsigned w = sample.width();
    unsigned h = sample.height();

    for (unsigned r = 0; r < h; r++)
        for (unsigned c = 0; c < w; c++)
            tensor.set(0, r, c, static_cast<float>(sample.getPixel(c, r)) / 255.0f);
}