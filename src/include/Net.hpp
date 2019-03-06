#ifndef CONNIE_NET_HPP
#define CONNIE_NET_HPP

#include <iostream>
#include <vector>
#include <memory>
#include "Tensor.hpp"
#include "LayerBase.hpp"

class Net
{
public:
    friend class TrainerBase;

    Net() = default;
    ~Net() = default;

    /**
     * Adds a layer to the network
     *
     * @param layer Shared pointer to the layer
     * @return      A reference to the network itself, to allow method chaining
     */
    Net &appendLayer(std::shared_ptr<LayerBase> layer);

    /**
     * Feeds the input tensor to the network and forwards it to compute the output
     */
    void forward();

    /**
     * Computes the gradient of the loss (forward() must have already been called)
     */
    void backward();

    /**
     * Gets a reference to the input tensor (useful to set the input data)
     *
     * @return
     */
    Tensor<> &getInput();

    /**
     * Gets a reference to the output tensor (useful to get regression data or classification probabilities)
     *
     * @return
     */
    Tensor<> &getOutput();

    /**
     * Set or unset training mode
     *
     * @param trainingMode
     */
    void setTrainingMode(bool trainingMode);

    /**
     * Returns the loss computed by calling backward()
     *
     * @return
     */
    float getLoss();

    /**
     * Save the network parameters into a file. Note that the the structure of the network is not saved.
     *
     * @param   filename
     * @return  True in case of success, otherwise false
     */
    bool save(const char *filename);

    /**
     * Save the network parameters into a stream. Note that the the structure of the network is not saved.
     *
     * @param   filename
     * @return  True in case of success, otherwise false
     */
    bool save(std::ostream &stream);

    /**
     * Loads the network parameters from a file. Note that the the structure of the network is not loaded.
     *
     * @param   filename
     * @return  True in case of success, otherwise false
     */
    bool load(const char *filename);

    /**
     * Loads the network parameters from a stream. Note that the the structure of the network is not loaded.
     *
     * @param   filename
     * @return  True in case of success, otherwise false
     */
    bool load(std::istream &stream);
private:
    /**
     * Shared pointers to the layers of the network
     */
    std::vector<std::shared_ptr<LayerBase>> layers;

    /**
     * Input tensor (allocated by the input layer)
     */
    Tensor<> *input;

    /**
     * Output tensor (allocated by the output layer)
     */
    Tensor<> *output;

    bool trainingMode = false;
};


#endif //CONNIE_NET_HPP
