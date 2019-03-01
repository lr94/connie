#include <catch2/catch.hpp>
#include "Net.hpp"
#include "InputLayer.hpp"
#include "ConvolutionalLayer.hpp"

TEST_CASE("Convolutional layer forward", "[convolution]")
{
    Net network;
    std::shared_ptr<ConvolutionalLayer> conv = std::make_shared<ConvolutionalLayer>(1, 3, 2, 1);
    network.appendLayer(std::make_shared<InputLayer>(3, 5, 5))
           .appendLayer(conv);

    REQUIRE(conv->output->depth() == 1);
    REQUIRE(conv->output->height() == 3);
    REQUIRE(conv->output->width() == 3);
}