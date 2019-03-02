#include "ConvolutionalLayer.hpp"

ConvolutionalLayer::ConvolutionalLayer(unsigned kernels, unsigned kernelSize, unsigned stride, unsigned padding)
    : ConvolutionalLayer(kernels, kernelSize, kernelSize, stride, stride, padding, padding) {}

ConvolutionalLayer::ConvolutionalLayer(unsigned kernels, unsigned kernelWidth, unsigned kernelHeight,
        unsigned strideX, unsigned strideY, unsigned paddingX, unsigned paddingY) : kernelCount(kernels),
        kernelWidth(kernelWidth), kernelHeight(kernelHeight), strideX(strideX), strideY(strideY), padX(paddingX),
        padY(paddingY)
{
    output = input = dOutput = dInput = nullptr;
}

ConvolutionalLayer::~ConvolutionalLayer()
{
    kernels.clear();
    delete output;
    delete dOutput;
}

void ConvolutionalLayer::forward()
{
    auto ih = static_cast<int>(inputHeight);
    auto iw = static_cast<int>(inputWidth);

    // For each kernel
    for (unsigned ki = 0; ki < kernelCount; ki++)
    {
        // For each "top left" of dot product (row-column)
        int yInput = -padY;
        int xInput;

        for (unsigned yOutput = 0; yOutput < outputHeight; yOutput++, yInput += strideY)
        {
            xInput = -padX;
            for (unsigned xOutput = 0; xOutput < outputWidth; xOutput++, xInput += strideX)
            {
                float sum = 0.0f;

                // For each kernel element (row-column-layer)
                for (unsigned i = 0; i < kernelHeight; i++)
                {
                    for (unsigned j = 0; j < kernelWidth; j++)
                    {
                        int yInput2 = yInput + i;
                        int xInput2 = xInput + j;

                        // If we are not in "padding area"
                        if (yInput2 >= 0 && yInput2 < ih && xInput2 >= 0 && xInput2 < iw)
                        {
                            auto yInput2U = static_cast<unsigned>(yInput2);
                            auto xInput2U = static_cast<unsigned>(xInput2);
                            for (unsigned l = 0; l < inputDepth; l++)
                                sum += kernels[ki].get(l, i, j) * input->get(l, yInput2U, xInput2U);

                        }
                    }
                }

                output->set(ki, yOutput, xOutput, sum + biases[ki]);
            }
        }
    }
}

void ConvolutionalLayer::backward()
{
    auto ih = static_cast<int>(inputHeight);
    auto iw = static_cast<int>(inputWidth);

    // Zero out input gradient
    dInput->zero();

    // For each kernel
    for (unsigned ki = 0; ki < kernelCount; ki++)
    {
        // For each "top left" of dot product (row-column)
        int yInput = -padY;
        int xInput;
        float dBiasSum = 0.0f;

        for (unsigned yOutput = 0; yOutput < outputHeight; yOutput++, yInput += strideY)
        {
            xInput = -padX;
            for (unsigned xOutput = 0; xOutput < outputWidth; xOutput++, xInput += strideX)
            {
                // Gradient from the next layer
                float outGrad = dOutput->get(ki, yOutput, xOutput);
                dBiasSum += outGrad;
                // For each kernel element (row-column-layer)
                for (unsigned i = 0; i < kernelHeight; i++)
                {
                    int yInput2 = yInput + i;
                    for (unsigned j = 0; j < kernelWidth; j++)
                    {
                        int xInput2 = xInput + j;

                        // If we are not in "padding area"
                        if (yInput2 >= 0 && yInput2 < ih && xInput2 >= 0 && xInput2 < iw)
                        {
                            auto yInput2U = static_cast<unsigned>(yInput2);
                            auto xInput2U = static_cast<unsigned>(xInput2);
                            for (unsigned l = 0; l < inputDepth; l++)
                            {
                                dKernels[ki].addAt(l, i, j, outGrad * input->get(l, yInput2U, xInput2U));
                                dInput->addAt(l, yInput2U, xInput2U, outGrad * kernels[ki].get(l, i, j));
                            }
                        }
                    }
                }
            }
        }

        dBiases[ki] += dBiasSum;
    }
}

void ConvolutionalLayer::updateParams(const TrainerBase &trainer)
{
    trainer.updateLayerParams(biases, dBiases);

    for (unsigned i = 0; i < kernelCount; i++)
        trainer.updateLayerParams(kernels[i], dKernels[i]);
}

void ConvolutionalLayer::prepend(LayerBase *previousLayer)
{
    LayerBase::prepend(previousLayer);

    inputHeight = input->height();
    inputWidth = input->width();
    inputDepth = input->depth();

    outputHeight = (inputHeight + 2 * padY - kernelHeight) / strideY + 1;
    outputWidth = (inputWidth + 2 * padX - kernelWidth) / strideX + 1;
    outputDepth = kernelCount;

    output = new Tensor<>(outputDepth, outputHeight, outputWidth);
    dOutput = new Tensor<>(outputDepth, outputHeight, outputWidth);

    kernels.clear();
    dKernels.clear();
    biases.clear();
    dBiases.clear();

    std::random_device r;
    std::default_random_engine generator(r());
    std::normal_distribution<float> distribution(0.0, 1.0);

    for (unsigned i = 0; i < kernelCount; i++)
    {
        kernels.emplace_back(Tensor<>::random(inputDepth, kernelHeight, kernelWidth));
        Tensor<> zeroTensor(inputDepth, kernelHeight, kernelWidth);
        zeroTensor.zero();
        dKernels.emplace_back(zeroTensor);
        biases.emplace_back(distribution(generator));
    }

    dBiases.insert(dBiases.end(), biases.size(), 0.0f);
}

bool ConvolutionalLayer::save(std::ostream &stream)
{
    for (auto &k : kernels)
    {
        size_t size = k.getDataSize();
        for (unsigned i = 0; i < size; i++)
            if (!writeFloat(stream, k.get(i)))
                return false;
    }

    for (auto &b : biases)
        if (!writeFloat(stream, b))
            return false;

    return true;
}

bool ConvolutionalLayer::load(std::istream &stream)
{
    for (auto &k : kernels)
    {
        size_t size = k.getDataSize();
        for (unsigned i = 0; i < size; i++)
        {
            float value;
            if (!readFloat(stream, value))
                return false;
            k.set(i, value);
        }
    }

    for (auto &b : biases)
        if (!readFloat(stream, b))
            return false;

    return true;
}