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
#include "ReluLayer.hpp"
#include "ConvolutionalLayer.hpp"
#include "MaxPoolingLayer.hpp"
#include "SoftmaxLayer.hpp"

#include "MnistDataset.hpp"

static void loadSample(Tensor<> &tensor, const MnistSample &sample);
void train(MnistDataset &dataset, Net &network, std::shared_ptr<SoftmaxLayer> &softmax,
           std::shared_ptr<TrainerBase> &trainer, std::string &logFile, std::string &preTrainedNetworkFile);
void test(MnistDataset &dataset, Net &network, std::shared_ptr<SoftmaxLayer> &softmax, unsigned examples);
void printHelp(char *argv[]);
Net initNetwork(std::shared_ptr<SoftmaxLayer> &softmaxLayer, int argc, char *argv[]);
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
    if (argc < 2)
    {
        printHelp(argv);
        return EXIT_SUCCESS;
    }

    std::string dataFile = readStringArg("data-file", defaultDataFile, argc, argv);
    std::string labelsFile = readStringArg("label-file", defaultLabelsFile, argc, argv);
    std::string preTrainedNetworkFile = readStringArg("network-file", defaultNetworkFile, argc, argv);
    std::string action = readStringArg("action", "train", argc, argv);

    std::cout << "Loading dataset from " << labelsFile << " and " << dataFile << std::endl;
    MnistDataset dataset(dataFile.c_str(), labelsFile.c_str());
    std::cout << "Loaded " << dataset.size() << " samples." << std::endl;

    // Build the network
    std::shared_ptr<SoftmaxLayer> softmax = std::make_shared<SoftmaxLayer>();
    Net network = initNetwork(softmax, argc, argv);

    // Load the network parameters
    if (std::filesystem::exists(preTrainedNetworkFile))
    {
        network.load(preTrainedNetworkFile.c_str());
        std::cout << "Loaded network from " << preTrainedNetworkFile << std::endl;
    }

    if (action == "test")
    {
        unsigned nExamples = static_cast<unsigned>(readIntArg("sample-size", 2500, argc, argv));
        test(dataset, network, softmax, nExamples);
    }
    else
    {
        if (action != "train")
            std::cout << "Unknown action \"" << action << R"(", falling back on "train")" << std::endl;

        std::string logFile = readStringArg("log", "", argc, argv);

        // Init the trainer
        std::shared_ptr<TrainerBase> trainer = initTrainer(network, argc, argv);

        train(dataset, network, softmax, trainer, logFile, preTrainedNetworkFile);
    }
}

void train(MnistDataset &dataset, Net &network, std::shared_ptr<SoftmaxLayer> &softmax,
        std::shared_ptr<TrainerBase> &trainer, std::string &logFile, std::string &preTrainedNetworkFile)
{
    network.setTrainingMode(true);

    Tensor<> &input = network.getInput();

    std::random_device random;
    std::mt19937 mersenne(random());

    // Init log file if necessary
    std::ofstream log;
    if (!logFile.empty())
    {
        log.open(logFile);
        log << "milliseconds,epoch,iteration,loss" << std::endl;
    }

    std::chrono::time_point start = std::chrono::system_clock::now();
    unsigned long long iteration = 0;
    for (unsigned epoch = 0; epoch < std::numeric_limits<unsigned>::max(); epoch++)
    {
        std::cout << "Shuffle dataset..." << std::endl;
        std::shuffle(dataset.begin(), dataset.end(), mersenne);
        for (auto &sample : dataset)
        {
            loadSample(input, sample);
            softmax->setTargetClass(sample.label());

            trainer->train();

            iteration++;
            if (iteration % (16 * 10) == 0)
            {
                float loss = trainer->getLoss();
                std::cout << "Epoch: " << epoch << " iteration: " << iteration << " loss: " << loss << std::endl;
                if (!logFile.empty())
                {
                    long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start).count();
                    log << ms << "," << epoch << "," << iteration << "," << loss << std::endl;
                }
                if (std::isnan(loss))
                    exit(0);
            }
            if (iteration % (16 * 10 * 10) == 0)
            {
                network.save(preTrainedNetworkFile.c_str());
                std::cout << "Network saved in " << preTrainedNetworkFile << std::endl;
            }
        }
    }
}

void test(MnistDataset &dataset, Net &network, std::shared_ptr<SoftmaxLayer> &softmax, unsigned examples)
{
    network.setTrainingMode(false);

    std::random_device random;
    std::mt19937 mersenne(random());

    Tensor<> &input = network.getInput();

    std::shuffle(dataset.begin(), dataset.end(), mersenne);
    unsigned ok = 0, tot = std::min<unsigned>(examples, static_cast<unsigned>(dataset.size()));
    for (unsigned i = 0; i < tot; i++)
    {
        MnistSample &sample = dataset[i];
        loadSample(input, sample);
        network.forward();

        if (softmax->getPredictedClass() == sample.label())
            ok++;
    }
    float accuracy = static_cast<float>(ok) / tot * 100.0f;
    std::cout << "Accuracy: " << std::setprecision(4) << accuracy << " %" << std::endl;
    std::cout << "Correct classifications: " << ok << std::endl;
    std::cout << "Errors: " << tot - ok << std::endl;
    std::cout << "Total: " << tot << std::endl;
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

std::shared_ptr<TrainerBase> initTrainer(Net &network, int argc, char *argv[])
{
    std::string trainerString = readStringArg("trainer", "sgd", argc, argv);

    float learningRate = readFloatArg("learning-rate", 0.01f, argc, argv);
    float momentum = readFloatArg("momentum", 0.9f, argc, argv);
    float decayRate = readFloatArg("decay-rate", 0.9, argc, argv);
    unsigned batchSize = static_cast<unsigned>(readIntArg("batch-size", 16, argc, argv));

    std::shared_ptr<TrainerBase> trainer;
    std::string trainerName;

    if (trainerString == "sgd")
    {
        trainer = std::make_shared<SGDTrainer>(network, learningRate, batchSize);
        trainerName = "Stochastic Gradient Descent";
    }
    else if (trainerString == "momentum")
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
        std::cout << "Unknown optimizer " << trainerString << std::endl;
        std::cout << "Valid optimizers are: sgd, momentm, nesterov, adagrad, rmsprop" << std::endl;
        exit(1);
    }

    std::cout << "Optimization algorithm: " << trainerName << std::endl;
    std::cout << "Learning rate: " << learningRate << std::endl;
    if (trainerString == "momentum" || trainerString == "nesterov" || trainerString == "rmsprop")
    {
        std::cout << "Momentum: " << momentum << std::endl;

        if (trainerString == "rmsprop")
            std::cout << "Decay rate: " << decayRate << std::endl;
    }
    std::cout << "Batch size: " << batchSize << std::endl;

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

void printHelp(char *argv[])
{
    std::cout << "Usage:" << std::endl;
    std::cout << "\t" << argv[0] << std::endl << std::endl;
    std::cout << "\t\t --action\t\tSelect the action to perform. Possible values: train, test" << std::endl;
    std::cout << "\t\t --sample-size\t\tSpecifies the number of examples to use for the test" << std::endl;
    std::cout << "\t\t --network-type\t\tSelect the type of network. Possible values: lenet or fc" << std::endl;
    std::cout << "\t\t --trainer\t\tSelect the optimizer. Possible values: sgd, momentum, nesterov, adagrad or rmsprop" << std::endl;
    std::cout << "\t\t --learning-rate\tThe learning rate. Default value: 0.01" << std::endl;
    std::cout << "\t\t --momentum\t\tSpecifies the momentum for optimizers supporting it. Default value: 0.9" << std::endl;
    std::cout << "\t\t --decay-rate\t\tDecay rate for RMSProp" << std::endl;
    std::cout << "\t\t --batch-size\t\tMinibatch size; default value: 16" << std::endl;
    std::cout << "\t\t --data-file\t\tPath of the MNIST dataset data file" << std::endl;
    std::cout << "\t\t --label-file\t\tPath of the MNIST dataset label file" << std::endl;
    std::cout << "\t\t --network-file\t\tPath of the file used to load and store the network weights. Default: network.bin" << std::endl;
    std::cout << "\t\t --log\t\t\tPath of a CSV file to store training progress. Default: none" << std::endl;
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