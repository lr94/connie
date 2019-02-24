#include <iostream>
#include <gd.h>
#include <SDL2/SDL.h>

#include "../main/Tensor.hpp"
#include "../main/Net.hpp"
#include "../main/SGDTrainer.hpp"
#include "../main/SigmoidLayer.hpp"
#include "../main/RegressionLayer.hpp"
#include "../main/FullyConnectedLayer.hpp"
#include "../main/ReluLayer.hpp"
#include "../main/InputLayer.hpp"

int main(int argc, char *argv[])
{
    gdImagePtr img = gdImageCreateFromFile(argv[1]);
    int width = gdImageSX(img);
    int height = gdImageSY(img);
    Tensor<> imageTensor(3, height, width);
    for (int x = 0; x < width; x++)
        for (int y = 0; y < height; y++)
        {
            int color = gdImageGetPixel(img, x, y);
            imageTensor[0][y][x] = static_cast<float>((color >> 16) & 0xff) / 255.0f;
            imageTensor[1][y][x] = static_cast<float>((color >> 8) & 0xff) / 255.0f;
            imageTensor[2][y][x] = static_cast<float>(color & 0xff) / 255.0f;
        }
    std::cout << width << "x" << height << std::endl;
    gdImageDestroy(img);

    InputLayer inputLayer(2, 1, 1);
    FullyConnectedLayer fc1(20);
    FullyConnectedLayer fc2(20);
    FullyConnectedLayer fc3(3);
    ReluLayer relu1;
    ReluLayer relu2;
    ReluLayer relu3;
    RegressionLayer regression;

    Net network;
    network.appendLayer(inputLayer)
           .appendLayer(fc1)
           .appendLayer(relu1)
           .appendLayer(fc2)
           .appendLayer(relu2)
           .appendLayer(fc3)
           .appendLayer(relu3)
           .appendLayer(regression);

    Tensor<> &input = network.getInput();
    Tensor<> &output = network.getOutput();
    Tensor<> &target = regression.target();

    // Initialize the trainer
    SGDTrainer trainer(0.001f);

    SDL_Renderer *renderer;
    SDL_Window *window;
    SDL_Init(SDL_INIT_VIDEO);
    SDL_CreateWindowAndRenderer(width, height, 0, &window, &renderer);

    unsigned i = 0;
    for (int k = 0; k < 1000000; k++)
    {
        for (unsigned y = 0; y < height; y++)
            for (unsigned x = 0; x < width; x++)
            {
                input.set(0, x);
                input.set(1, y);
                target.set(0, imageTensor.get(0, y, x));
                target.set(1, imageTensor.get(1, y, x));
                target.set(2, imageTensor.get(2, y, x));

                network.train(trainer);

                if (i++ % 1000 == 0)
                    std::cout << network.getLoss() << std::endl;
            }

        float loss = network.getLoss();
        std::cout << loss << std::endl;

        if (k % 1000 == 0)
        {
            for (unsigned y = 0; y < height; y++)
                for (unsigned x = 0; x < width; x++)
                {
                    input.set(0, x);
                    input.set(1, y);

                    network.forward();

                    auto r = static_cast<uint8_t>(output.get(0) * 255);
                    auto g = static_cast<uint8_t>(output.get(1) * 255);
                    auto b = static_cast<uint8_t>(output.get(2) * 255);

                    // std::cout << "(" << x << " , " << y << ") -> " << output << std::endl;

                    SDL_SetRenderDrawColor(renderer, r, g, b, 0xff);
                    SDL_RenderDrawPoint(renderer, x, y);
                }
            SDL_RenderPresent(renderer);
        }
    }

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
}