#include <limits>
#include "MaxPoolingLayer.hpp"

MaxPoolingLayer::MaxPoolingLayer(unsigned windowSize, unsigned stride, unsigned padding)
    : PoolingLayerBase(windowSize, stride, padding) {}

MaxPoolingLayer::MaxPoolingLayer(unsigned windowWidth, unsigned windowHeight, unsigned strideX, unsigned strideY,
                unsigned paddingX, unsigned paddingY) : PoolingLayerBase(windowWidth, windowHeight, strideX, strideY,
                        paddingX, paddingY) {}

MaxPoolingLayer::~MaxPoolingLayer()
{
    delete output;
    delete dOutput;
}

void MaxPoolingLayer::forward()
{
    auto ih = static_cast<int>(inputHeight);
    auto iw = static_cast<int>(inputWidth);

    int yInput = -padY;
    int xInput;

    // For each "top-left" element of the window
    for (unsigned yOutput = 0; yOutput < outputHeight; yOutput++, yInput += strideY)
    {
        xInput = -padX;
        for (unsigned xOutput = 0; xOutput < outputWidth; xOutput++, xInput += strideX)
        {
            for (unsigned l = 0; l < inputDepth; l++)
            {
                float max = -std::numeric_limits<float>::infinity();

                // For each window element (row-column-layer)
                for (unsigned i = 0; i < windowHeight; i++)
                {
                    for (unsigned j = 0; j < windowWidth; j++)
                    {
                        int yInput2 = yInput + i;
                        int xInput2 = xInput + j;

                        float val = 0.0f;
                        // If we are not in "padding area"
                        if (yInput2 >= 0 && yInput2 < ih && xInput2 >= 0 && xInput2 < iw)
                            val = input->get(l, static_cast<unsigned>(yInput2), static_cast<unsigned>(xInput2));

                        if (val > max)
                            max = val;
                    }
                }

                output->set(l, yOutput, xOutput, max);
            }
        }
    }
}

void MaxPoolingLayer::backward()
{

}