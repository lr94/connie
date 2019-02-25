#include <iostream>

#include "Dataset.hpp"

int main(int argc, char *argv[])
{
    std::cout << "Hello!" << std::endl;
    Dataset dataset("../datasets/mnist/train-images-idx3-ubyte", "../datasets/mnist/train-labels-idx1-ubyte");

    std::cout << dataset.size() << std::endl;
}