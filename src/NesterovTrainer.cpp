#include "NesterovTrainer.hpp"

NesterovTrainer::NesterovTrainer(Net &network, float learningRate, float momentum, unsigned batchSize)
        : TrainerBase(network, batchSize, 1), learningRate(learningRate), momentum(momentum) {}

void NesterovTrainer::updateLayerParams(std::vector<float> &params, std::vector<float> &gradient, std::vector<std::vector<float>> &addMem) const
{
    size_t size = params.size();

    std::vector<float> &v = addMem[0];

    for (unsigned i = 0; i < size; i++)
    {
        /*
         * Source:
         *      https://github.com/keras-team/keras/blob/master/keras/optimizers.py#L200
         * I'm not sure I understood this, possible explainations:
         *      - https://jlmelville.github.io/mize/nesterov.html
         *      - https://stackoverflow.com/questions/50774683/how-is-nesterovs-accelerated-gradient-descent-implemented-in-tensorflow
         *      - http://cs231n.github.io/neural-networks-3/#sgd
         *      - https://stats.stackexchange.com/questions/179915/whats-the-difference-between-momentum-based-gradient-descent-and-nesterovs-acc/233430#233430
         */
        float val = learningRate * gradient[i] / batchSize;
        float vi = momentum * v[i] - val;
        v[i] = vi;
        params[i] += momentum * vi - val;
    }

    // Zero out the gradient (end of minibatch)
    for (auto &g : gradient)
        g = 0;
}

void NesterovTrainer::updateLayerParams(Tensor<> &params, Tensor<> &gradient, std::vector<Tensor<>> &addMem) const
{
    size_t size = params.getDataSize();

    Tensor<> &v = addMem[0];

    for (unsigned i = 0; i < size; i++)
    {
        float val = learningRate * gradient.get(i) / batchSize;
        float vi = momentum * v.get(i) - val;
        v.set(i, vi);
        params.addAt(i, momentum * vi - val);
    }

    // Zero out the gradient (end of minibatch)
    gradient.zero();
}