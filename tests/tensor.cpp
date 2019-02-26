#define CATCH_CONFIG_MAIN

#include <catch2/catch.hpp>

#include "Tensor.hpp"

TEST_CASE("Tensor initialization", "[tensor][init]")
{
    const unsigned depth = 2;
    const unsigned height = 3;
    const unsigned width = 4;

    Tensor<> tensor(depth, height, width);

    SECTION("Tensor size")
    {
        REQUIRE(tensor.getDataSize() == depth * height * width);
    }

    SECTION("Tensor dimensions")
    {
        REQUIRE(tensor.depth() == depth);
        REQUIRE(tensor.height() == height);
        REQUIRE(tensor.width() == width);
    }

    size_t size = tensor.getDataSize();
    for (size_t i = 0; i < size; i++)
        tensor.set(i, i + 1);

    SECTION("Check tensor values")
    {
        size_t size = tensor.getDataSize();
        for (size_t i = 0; i < size; i++)
            REQUIRE(tensor.get(i) == i + 1);
    }

    SECTION("Check access by indices")
    {
        for (unsigned i = 0; i < depth; i++)
            for (unsigned j = 0; j < height; j++)
                for (unsigned k = 0; k < width; k++)
                    REQUIRE(tensor.get(i, j, k) == i * width * height + j * width + k + 1);
    }

    SECTION("Check access by [] operator")
    {
        for (unsigned i = 0; i < depth; i++)
            for (unsigned j = 0; j < height; j++)
                for (unsigned k = 0; k < width; k++)
                    REQUIRE(tensor[i][j][k] == i * width * height + j * width + k + 1);
    }
}