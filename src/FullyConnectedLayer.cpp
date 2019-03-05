#include "FullyConnectedLayer.hpp"
#include "TrainerBase.hpp"

FullyConnectedLayer::FullyConnectedLayer(unsigned numNeurons)
{
    output = new Tensor<>(numNeurons, 1, 1);
    dOutput = new Tensor<>(numNeurons, 1, 1);
    input = nullptr;
}

FullyConnectedLayer::~FullyConnectedLayer()
{
    weights.clear();
    delete output;
    delete dOutput;
}

void FullyConnectedLayer::forward()
{
    unsigned neurons = numNeurons();
    for (unsigned i = 0; i < neurons; i++)
    {
        float value = *input * weights[i] + biases[i];
        output->set(i, value); // Possible because width and height are 1, otherwise we should have used [i][0][0]
    }
}

void FullyConnectedLayer::backward()
{
    size_t inputSize = input->getDataSize();
    unsigned neurons = numNeurons();

    for (unsigned i = 0; i < neurons; i++)
    {
        float g_i = dOutput->get(i);

        // Compute gradient w.r.t. bias
        dBiases[i] += g_i; // dB = gradient

        Tensor<> &dNeuronWeights = dWeights[i];

        // Compute gradient w.r.t. weights
        for (unsigned j = 0; j < inputSize; j++) // dW = gradient * h
            dNeuronWeights.addAt(j, g_i * input->get(j));
    }

    // Compute gradient w.r.t. input
    for (unsigned j = 0; j < inputSize; j++)
    {
        float sum = 0;
        for (unsigned i = 0; i < neurons; i++)
            sum += dOutput->get(i) * weights[i].get(j); //dX = W * gradient
        dInput->set(j, sum);
    }
}

void FullyConnectedLayer::updateParams(const TrainerBase &trainer)
{
    trainer.updateLayerParams(biases, dBiases, additionalMemBiases);

    unsigned neurons = numNeurons();
    for (unsigned i = 0; i < neurons; i++)
        trainer.updateLayerParams(weights[i], dWeights[i], additionalMemWeights[i]);
}

inline unsigned FullyConnectedLayer::numNeurons() const
{
    return output->depth();
}

void FullyConnectedLayer::prepend(LayerBase *previousLayer)
{
    LayerBase::prepend(previousLayer);

    // Clear weights and biases
    weights.clear();
    dWeights.clear();
    biases.clear();
    dBiases.clear();

    unsigned neurons = numNeurons();

    // Weight initialization suggested by Xavier Glorot and Yoshua Bengio, 2010
    std::random_device r;
    std::default_random_engine generator(r());
    float boundary = std::sqrt(6.0f / ((input->depth() * input->height() * input->width()) + neurons));
    std::uniform_real_distribution<float> distribution(-boundary, boundary);

    // For each output unit
    for (unsigned i = 0; i < neurons; i++)
    {
        weights.emplace_back(Tensor<>::random(input->depth(), input->height(), input->width(), generator, distribution));
        Tensor<> zeroTensor(input->depth(), input->height(), input->width());
        zeroTensor.zero();
        dWeights.emplace_back(zeroTensor);
        biases.emplace_back(distribution(generator));
    }

    dBiases.insert(dBiases.end(), biases.size(), 0.0f);
}

void FullyConnectedLayer::initAdditionalMemory(unsigned additionalMemory)
{
    unsigned n = numNeurons();

    // For each unit
    additionalMemWeights.clear();
    for (unsigned i = 0; i < n; i++)
    {
        // Add the amount of additional memory slots required
        Tensor<> zeroTensor(input->depth(), input->height(), input->width());
        zeroTensor.zero();
        additionalMemWeights.emplace_back(std::vector<Tensor<>>(additionalMemory, zeroTensor));
    }

    additionalMemBiases.clear();
    // For each additional memory slot
    for (unsigned i = 0; i < additionalMemory; i++)
        additionalMemBiases.emplace_back(std::vector<float>(n, 0.0f));
}

bool FullyConnectedLayer::save(std::ostream &stream)
{
    for (auto &w : weights)
    {
        size_t size = w.getDataSize();
        for (unsigned i = 0; i < size; i++)
            if (!writeFloat(stream, w.get(i)))
                return false;
    }

    for (auto &b : biases)
        if (!writeFloat(stream, b))
            return false;

    return true;
}

bool FullyConnectedLayer::load(std::istream &stream)
{
    for (auto &w : weights)
    {
        size_t size = w.getDataSize();
        for (unsigned i = 0; i < size; i++)
        {
            float value;
            if (!readFloat(stream, value))
                return false;
            w.set(i, value);
        }
    }

    for (auto &b : biases)
        if (!readFloat(stream, b))
            return false;

    return true;
}