#ifndef CNN_LAYER_HPP
#define CNN_LAYER_HPP

#include <iostream>
#include "Tensor.hpp"
class TrainerBase;

/**
 * Each layer is responsible of its output tensor, the input one is the output tensor of the previous layer
 */
class LayerBase
{
public:
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
     * the parameters and their gradients (it can be called multiple times). The trainer will update the prameters.
     *
     * @param trainer
     */
    virtual void updateParams(const TrainerBase &trainer);

    void append(LayerBase *nextLayer);

    virtual bool save(std::ostream &stream);
    virtual bool load(std::istream &stream);

protected:
    virtual void prepend(LayerBase *previousLayer);

    bool writeFloat(std::ostream &stream, float value);

    bool readFloat(std::istream &stream, float &value);
};
#endif
