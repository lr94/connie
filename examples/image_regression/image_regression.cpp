#include <iostream>
#include <random>
#include <memory>
#include <csignal>
#include <gd.h>
#include <SDL2/SDL.h>

#include "Tensor.hpp"
#include "Net.hpp"
#include "SGDTrainer.hpp"
#include "SigmoidLayer.hpp"
#include "RegressionLayer.hpp"
#include "FullyConnectedLayer.hpp"
#include "ReluLayer.hpp"
#include "TanhLayer.hpp"
#include "InputLayer.hpp"

static float truncate(float x);

int main(int argc, char *argv[])
{
    gdImagePtr img = gdImageCreateFromFile(argv[1]);
    auto width = static_cast<unsigned>(gdImageSX(img));
    auto height = static_cast<unsigned>(gdImageSY(img));
    Tensor<> imageTensor(3, height, width);
    for (unsigned x = 0; x < width; x++)
        for (unsigned y = 0; y < height; y++)
        {
            int color = gdImageGetPixel(img, x, y);
            imageTensor[0][y][x] = static_cast<float>((color >> 16) & 0xff) / 255.0f;
            imageTensor[1][y][x] = static_cast<float>((color >> 8) & 0xff) / 255.0f;
            imageTensor[2][y][x] = static_cast<float>(color & 0xff) / 255.0f;
        }
    std::cout << width << "x" << height << std::endl;
    gdImageDestroy(img);

    std::shared_ptr<RegressionLayer> regression = std::make_shared<RegressionLayer>();

    Net network;
    network.appendLayer(std::make_shared<InputLayer>(2, 1, 1))
            .appendLayer(std::make_shared<FullyConnectedLayer>(20))
            .appendLayer(std::make_shared<ReluLayer>())
            .appendLayer(std::make_shared<FullyConnectedLayer>(20))
            .appendLayer(std::make_shared<ReluLayer>())
            .appendLayer(std::make_shared<FullyConnectedLayer>(20))
            .appendLayer(std::make_shared<ReluLayer>())
            .appendLayer(std::make_shared<FullyConnectedLayer>(20))
            .appendLayer(std::make_shared<ReluLayer>())
            .appendLayer(std::make_shared<FullyConnectedLayer>(20))
            .appendLayer(std::make_shared<ReluLayer>())
            .appendLayer(std::make_shared<FullyConnectedLayer>(20))
            .appendLayer(std::make_shared<ReluLayer>())
            .appendLayer(std::make_shared<FullyConnectedLayer>(20))
            .appendLayer(std::make_shared<ReluLayer>())
            .appendLayer(std::make_shared<FullyConnectedLayer>(3))
            .appendLayer(regression);

    Tensor<> &input = network.getInput();
    Tensor<> &output = network.getOutput();
    Tensor<> &target = regression->target();

    // Initialize the trainer
    SGDTrainer trainer(network, 0.01f, 5);

    SDL_Renderer *renderer;
    SDL_Window *window;
    SDL_Init(SDL_INIT_VIDEO);
    SDL_CreateWindowAndRenderer(width, height, 0, &window, &renderer);

    std::random_device random_device;
    std::mt19937 engine(random_device());
    std::uniform_int_distribution<> x_distr(0, width - 1);
    std::uniform_int_distribution<> y_distr(0, height - 1);

    signal(SIGINT, exit);

    for (int k = 0; k < 1000000000; k++)
    {
        auto x = static_cast<unsigned>(x_distr(engine));
        auto y = static_cast<unsigned>(y_distr(engine));

        float xfloat = (static_cast<float>(x) - static_cast<float>(width) / 2) / width;
        float yfloat = (static_cast<float>(y) - static_cast<float>(height) / 2) / height;

        input.set(0, xfloat);
        input.set(1, yfloat);
        target.set(0, imageTensor.get(0, y, x));
        target.set(1, imageTensor.get(1, y, x));
        target.set(2, imageTensor.get(2, y, x));

        trainer.train();

        if (k % 5000 == 0)
        {
            if (k % 25000 == 0)
            {
                float loss = trainer.getLoss();
                std::cout << loss << std::endl;
            }

            for (y = 0; y < height; y++)
                for (x = 0; x < width; x++)
                {
                    xfloat = (static_cast<float>(x) - static_cast<float>(width) / 2) / width;
                    yfloat = (static_cast<float>(y) - static_cast<float>(height) / 2) / height;

                    input.set(0, xfloat);
                    input.set(1, yfloat);

                    network.forward();

                    auto r = static_cast<uint8_t>(truncate(output.get(0) * 255));
                    auto g = static_cast<uint8_t>(truncate(output.get(1) * 255));
                    auto b = static_cast<uint8_t>(truncate(output.get(2) * 255));

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

static float truncate(float x)
{
    if (x < 0)
        return 0;

    if (x > 255)
        return 255;

    return x;
}