#include <iostream>
#include <iomanip>
#include <fstream>
#include <filesystem>
#include <memory>
#include <chrono>
#include <algorithm>
#include <random>

#include "Tensor.hpp"
#include "Net.hpp"

#include "SGDTrainer.hpp"
#include "MomentumTrainer.hpp"
#include "NesterovTrainer.hpp"
#include "AdaGradTrainer.hpp"
#include "RMSPropTrainer.hpp"

#include "InputLayer.hpp"
#include "FullyConnectedLayer.hpp"
#include "TanhLayer.hpp"
#include "ReluLayer.hpp"
#include "ConvolutionalLayer.hpp"
#include "MaxPoolingLayer.hpp"
#include "SigmoidLayer.hpp"
#include "SoftmaxLayer.hpp"

#include "MnistDataset.hpp"

static void loadSample(Tensor<> &tensor, const MnistSample &sample);
std::shared_ptr<TrainerBase> initTrainer(Net &network, int argc, char *argv[]);

float readFloatArg(const std::string &argumentName, float defaultValue, int argc, char *argv[]);
int readIntArg(const std::string &argumentName, int defaultValue, int argc, char *argv[]);
std::string readStringArg(const std::string &argumentName, const std::string &defaultValue, int argc, char *argv[]);

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
        else if (current_arg == "-bs" && next_arg_index < argc)
            batchSize = static_cast<unsigned int>(std::stoi(argv[++i]));
        else if (current_arg == "-l" && next_arg_index < argc)
            logFile = argv[++i];
    }

    std::cout << "Loading dataset from " << labelsFile << " and " << dataFile << std::endl;
    MnistDataset dataset(dataFile, labelsFile);
    std::cout << "Loaded " << dataset.size() << " samples." << std::endl;

    // Build the network
    Net network;
    std::shared_ptr<SoftmaxLayer> softmax = std::make_shared<SoftmaxLayer>();
    network.appendLayer(std::make_shared<InputLayer>(1, size, size))
            .appendLayer(std::make_shared<ConvolutionalLayer>(6, 5, 1, 2))
            .appendLayer(std::make_shared<ReluLayer>())
            .appendLayer(std::make_shared<MaxPoolingLayer>(2, 2, 0))
            .appendLayer(std::make_shared<ConvolutionalLayer>(16, 5, 1, 0))
            .appendLayer(std::make_shared<ReluLayer>())
            .appendLayer(std::make_shared<MaxPoolingLayer>(2, 2, 0))
            .appendLayer(std::make_shared<ConvolutionalLayer>(120, 1, 1, 0))
            .appendLayer(std::make_shared<ReluLayer>())
            .appendLayer(std::make_shared<FullyConnectedLayer>(84))
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

    std::random_device random;
    std::mt19937 mersenne(random());

    std::shuffle(dataset.begin(), dataset.end(), mersenne);
    unsigned ok = 0, tot = std::min<unsigned>(2500, static_cast<unsigned>(dataset.size()));
    for (unsigned i = 0; i < tot; i++)
    {
        MnistSample &sample = dataset[i];
        loadSample(input, sample);
        network.forward();

        if (softmax->getPredictedClass() == sample.label())
            ok++;
    }
    float accuracy = static_cast<float>(ok) / tot * 100.0f;
    std::cout << "Initial accuracy: " << std::setprecision(4) << accuracy << " %" << std::endl;

    // Init the trainer
    std::shared_ptr<TrainerBase> trainer = initTrainer(network, argc, argv);

    // Init log file if necessary
    std::ofstream log;
    if (logFile != nullptr)
    {
        log.open(logFile);
        log << "milliseconds,epoch,iteration,loss" << std::endl;
    }

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

            trainer->train();

            iteration++;
            if (iteration % (batchSize * 10) == 0)
            {
                float loss = trainer->getLoss();
                std::cout << "Epoch: " << epoch << " iteration: " << iteration << " loss: " << loss << std::endl;
                if (logFile != nullptr)
                {
                    long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start).count();
                    log << ms << "," << epoch << "," << iteration << "," << loss << std::endl;
                }
                if (std::isnan(loss))
                    exit(0);
            }
            if (iteration % (batchSize * 10 * 10) == 0)
            {
                network.save(networkFile);
                std::cout << "Network saved in " << networkFile << std::endl;
            }
        }
    }
}

std::shared_ptr<TrainerBase> initTrainer(Net &network, int argc, char *argv[])
{
    std::string trainerString = readStringArg("trainer", "sgd", argc, argv);

    float learningRate = readFloatArg("learning-rate", 0.01f, argc, argv);
    float momentum = readFloatArg("momentum", 0.9f, argc, argv);
    float decayRate = readFloatArg("decay-rate", 0.9, argc, argv);
    unsigned batchSize = static_cast<unsigned>(readIntArg("batch-size", 16, argc, argv));

    std::shared_ptr<TrainerBase> trainer;
    std::string trainerName;

    if (trainerString == "momentum")
    {
        trainer = std::make_shared<MomentumTrainer>(network, learningRate, momentum, batchSize);
        trainerName = "Stochastic Gradient Descent with momentum";
    }
    else if (trainerString == "nesterov")
    {
        trainer = std::make_shared<NesterovTrainer>(network, learningRate, momentum, batchSize);
        trainerName = "Nesterov Accelerated Gradient";
    }
    else if (trainerString == "adagrad")
    {
        trainer = std::make_shared<AdaGradTrainer>(network, learningRate, batchSize);
        trainerName = "AdaGrad";
    }
    else if (trainerString == "rmsprop")
    {
        trainer = std::make_shared<RMSPropTrainer>(network, learningRate, decayRate, momentum, batchSize);
        trainerName = "RMSProp";
    }
    else
    {
        trainer = std::make_shared<SGDTrainer>(network, learningRate, batchSize);
        trainerName = "Stochastic Gradient Descent";
    }

        std::cout << "Optimization algorithm: " << trainerName << std::endl;
    std::cout << "Learning rate: " << learningRate << std::endl;
    if (trainerString == "momentum" || trainerString == "nesterov" || trainerString == "rmsprop")
    {
        std::cout << "Momentum: " << momentum << std::endl;

        if (trainerString == "rmsprop")
            std::cout << "Decay rate: " << decayRate << std::endl;
    }

    return trainer;
}

static void loadSample(Tensor<> &tensor, const MnistSample &sample)
{
    unsigned w = sample.width();
    unsigned h = sample.height();

    for (unsigned r = 0; r < h; r++)
        for (unsigned c = 0; c < w; c++)
            tensor.set(0, r, c, static_cast<float>(sample.getPixel(c, r)) / 255.0f);
}

float readFloatArg(const std::string &argumentName, float defaultValue, int argc, char *argv[])
{
    std::string fullName = "--" + argumentName;

    for (int i = 1; i < argc; i++)
    {
        int next_arg_index = i + 1;
        std::string current_arg = argv[i];

        if (current_arg == fullName && next_arg_index < argc)
            return std::stof(argv[++i]);
    }

    return defaultValue;
}

int readIntArg(const std::string &argumentName, int defaultValue, int argc, char *argv[])
{
    std::string fullName = "--" + argumentName;

    for (int i = 1; i < argc; i++)
    {
        int next_arg_index = i + 1;
        std::string current_arg = argv[i];

        if (current_arg == fullName && next_arg_index < argc)
            return std::stoi(argv[++i]);
    }

    return defaultValue;
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