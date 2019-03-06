#ifndef CONNIE_LAYER_HPP
#define CONNIE_LAYER_HPP

#include <iostream>
#include "Tensor.hpp"
class TrainerBase;

/**
 * Each layer is responsible of its output tensor, the input one is the output tensor of the previous layer
 */
class LayerBase
{
public:
    friend class TrainerBase;
    friend class Net;

    /**
     * Input tensor of the layer (allocated by the previous layer, with the InputLayer exception)
     */
    Tensor<> *input;
    /**
     * Output tensor of the layer (allocated by this layer when prepend() is called)
     */
    Tensor<> *output;

    /**
     * Gradient tensor of the loss w.r.t. the input tensor. Allocated by the previous layer
     */
    Tensor<> *dInput;
    /**
     * Gradient tensor of the loss w.r.t. the output tensor. Allocated by this layer
     */
    Tensor<> *dOutput;

    /**
     * Takes the data coming from the previous layer and computes the output tensor. The layer must have a previous
     * layer (with the exception of the InputLayer).
     *
     * This is a pure virtual method, so it must be overridden
     */
    virtual void forward() = 0;

    /**
     * Computes the gradient of loss w.r.t. the layer parameters (if any) and the input tensor. The layer must have a
     * previous layer (with the except of the InputLayer).
     * Note that while the gradient w.r.t. the input is cleared every iteration, the gradient w.r.t. the parameters
     * gets accumulated for stochastic gradient descent purposes. It will be cleared by the trainer at the end
     * of every batch.
     *
     * This is a pure virtual method, so it must be overridden
     */
    virtual void backward() = 0;

    /**
     * This method is called by the trainer (instance of TrainerBase) which pass itself as an argument.
     * The default implementation doesn't do anything, but layers with trainable parameters need to override it.
     * When this method is invoked the layer should invoke the method updateLayerParams() of the trainer, passing
     * the parameters and their gradients (it can be called multiple times). The trainer will update the prameters and
     * if necessary zero out the gradient w.r.t them
     *
     * @param trainer The trainer, obviously
     */
    virtual void updateParams(const TrainerBase &trainer);

    /**
     * Add a layer after this one. Internally it works calling prepend() on the next layer
     *
     * @param nextLayer The layer to be added
     */
    void append(LayerBase *nextLayer);

    /**
     * This method allows layers to save their parameters into a binary stream (this should be done using writeFloat())
     * The default implementation doesn't do anything.
     *
     * @param stream The stream
     * @return       True in case of success, otherwise false
     */
    virtual bool save(std::ostream &stream);

    /**
     * This method allows the layer to load its parameters previously saved with save(). It should use readFloat() and
     * read the exact number of bytes written by save()
     *
     * @param stream The stream
     * @return       True in case of success, otherwise false
     */
    virtual bool load(std::istream &stream);

protected:
    /**
     * Appends this layer to a specified one. This method can be overriden to allow layers to prepare their structures
     * containing parameters, but any implementation should always call the parent method
     *
     * @param previousLayer
     */
    virtual void prepend(LayerBase *previousLayer);

    /**
     * Allows the trainer (this method is protected but it's alled by TrainerBase which is a friend) to setup additional
     * parameters useful for the specific optimization algorithm. The default implementation doesn't do anything.
     *
     * @param additionalMemory Number of extra float values to be stored for each parameter
     */
    virtual void initAdditionalMemory(unsigned additionalMemory);

    /**
     * Writes a float into the stream (it should do it with little endian byte order, but for now it is machine dependent)
     *
     * @param stream
     * @param value
     * @return       True in case of success, otherwise false
     */
    bool writeFloat(std::ostream &stream, float value);

    /**
     * Reads a float from the stream (it should do it with little endian byte order, but for now it is machine dependent)
     *
     * @param stream
     * @param value
     * @return       True in case of success, otherwise false
     */
    bool readFloat(std::istream &stream, float &value);

    bool trainingMode = false;
};
#endif
