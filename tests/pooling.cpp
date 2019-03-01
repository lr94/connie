#include <catch2/catch.hpp>
#include "Net.hpp"
#include "InputLayer.hpp"
#include "MaxPoolingLayer.hpp"
#include "MemoryStream.hpp"

TEST_CASE("Max pooling", "[pooling][max]")
{
    Net network;

    SECTION("Test 1")
    {
        std::shared_ptr<MaxPoolingLayer> pooling = std::make_shared<MaxPoolingLayer>(2, 2, 0);
        network.appendLayer(std::make_shared<InputLayer>(1, 4, 4))
                .appendLayer(pooling);

        network.getInput() = {{1, 1, 2, 4}, {5, 6, 7, 8}, {3, 2, 1, 0}, {1, 2, 3, 4}};
        network.forward();
        Tensor<> expectedResult(1, 2, 2);
        expectedResult = {{6, 8}, {3, 4}};
        REQUIRE(network.getOutput() == expectedResult);
    }

    SECTION("Test 2")
    {
        std::shared_ptr<MaxPoolingLayer> pooling = std::make_shared<MaxPoolingLayer>(3, 1, 1);
        network.appendLayer(std::make_shared<InputLayer>(2, 3, 3))
                .appendLayer(pooling);

        network.getInput() = {{{1, -2, 3}, {2, -1, 1}, {4, -3, -4}}, {{6, -8, -1}, {-1, -3, -2}, {-2, 1, 0}}};
        network.forward();
        Tensor<> expectedResult(2, 3, 3);
        expectedResult = {{{2, 3, 3}, {4, 4, 3}, {4, 4, 1}}, {{6, 6, 0}, {6, 6, 1}, {1, 1, 1}}};
        REQUIRE(network.getOutput() == expectedResult);
    }
}