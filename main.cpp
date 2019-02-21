#include <iostream>
#include "Vol.hpp"
#include "InputLayer.hpp"
#include "FullyConnectedLayer.hpp"

int main()
{
    Vol<> v(5, 1, 1);
    v = {1, 2, 3, 4, 5};

    InputLayer inputLayer(5, 1, 1);
    FullyConnectedLayer fcc(5);

    inputLayer.append(&fcc);

    (*inputLayer.input) = v;

    fcc.forward();

    std::cout << (*fcc.output) << std::endl;

    return 0;
}