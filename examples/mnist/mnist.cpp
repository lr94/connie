#include <iostream>
#include <fstream>
#include <filesystem>
#include <memory>
#include <chrono>
#include <algorithm>
#include <random>

#include "Tensor.hpp"
#include "Net.hpp"
#include "SGDTrainer.hpp"
#include "InputLayer.hpp"
#include "FullyConnectedLayer.hpp"
#include "TanhLayer.hpp"
#include "ReluLayer.hpp"
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
    float learningRate = 0.005f;
    unsigned batchSize = 8;

    if (argc < 2)
        std::cout << "Usage:\n\t" << argv[0] << " [-n network_file] [-tl training_labels_file]"
                                                " [-td training_data_file] [-l log_file]"
                                                " [-lr learning_rate] [-bs batch_size]" << std::endl << std::endl;
    for (int i = 1; i < argc; i++)
    {
        int next_arg_index = i + 1;
        std::string current_arg = argv[i];

        if (current_arg == "-n" && next_arg_index < argc)
            networkFile = argv[++i];
        else if (current_arg == "-tl" && next_arg_index < argc)
            labelsFile = argv[++i];
        else if (current_arg == "-td" && next_arg_index < argc)
            dataFile = argv[++i];
        else if (current_arg == "-l" && next_arg_index < argc)
            logFile = argv[++i];
        else if (current_arg == "-lr" && next_arg_index < argc)
            learningRate = std::stof(argv[++i]);
        else if (current_arg == "-bs" && next_arg_index < argc)
            batchSize = static_cast<unsigned int>(std::stoi(argv[++i]));
        else if (current_arg == "-l" && next_arg_index < argc)
            logFile = argv[++i];
    }

    std::cout << "Loading dataset from " << labelsFile << " and " << dataFile << std::endl;
    Dataset dataset(dataFile, labelsFile);
    std::cout << "Loaded " << dataset.size() << " samples." << std::endl;

    // Build the network
    Net network;
    std::shared_ptr<SoftmaxLayer> softmax = std::make_shared<SoftmaxLayer>();
    network.appendLayer(std::make_shared<InputLayer>(1, size, size))
            .appendLayer(std::make_shared<FullyConnectedLayer>(size * size * 2))
            .appendLayer(std::make_shared<ReluLayer>())
            .appendLayer(std::make_shared<FullyConnectedLayer>(size * size))
            .appendLayer(std::make_shared<ReluLayer>())
            .appendLayer(std::make_shared<FullyConnectedLayer>(10))
            .appendLayer(softmax);
    // Load the network parameters
    if (std::filesystem::exists(networkFile))
    {
        network.load(networkFile);
        std::cout << "Loaded network from " << networkFile << std::endl;
    }
    Tensor<> &input = network.getInput();
    // Tensor<> &output = network.getOutput();

    // Init the trainer
    std::cout << "Learning rate: " << learningRate << std::endl << "Batch size: " << batchSize << std::endl;
    SGDTrainer trainer(network, learningRate, batchSize);

    // Init log file if necessary
    std::ofstream log;
    if (logFile != nullptr)
    {
        log.open(logFile);
        log << "milliseconds,epoch,iteration,loss" << std::endl;
    }

    std::random_device random;
    std::mt19937 mersenne(random());

    std::chrono::time_point start = std::chrono::system_clock::now();
    unsigned long long iteration = 0;
    for (unsigned epoch = 0; epoch < 1000000; epoch++)
    {
        std::cout << "Shuffle dataset..." << std::endl;
        std::shuffle(dataset.begin(), dataset.end(), mersenne);
        for (auto &sample : dataset)
        {
            loadSample(input, sample);
            softmax->setTargetClass(sample.label());

            trainer.train();

            iteration++;
            if (iteration % (batchSize * 1) == 0)
            {
                float loss = trainer.getLoss();
                std::cout << "Epoch: " << epoch << " iteration: " << iteration << " loss: " << loss << std::endl;
                if (logFile != nullptr)
                {
                    long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start).count();
                    log << ms << "," << epoch << "," << iteration << "," << loss << std::endl;
                }
                if (std::isnan(loss))
                    exit(0);
            }
            if (iteration % (batchSize * 1 * 10) == 0)
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