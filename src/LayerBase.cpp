#include "LayerBase.hpp"

void LayerBase::append(LayerBase *nextLayer)
{
    nextLayer->prepend(this);
}

bool LayerBase::save(std::ostream &stream)
{
    return true;
}

bool LayerBase::load(std::istream &stream)
{
    return true;
}

void LayerBase::prepend(LayerBase *previousLayer)
{
    input = previousLayer->output;
    dInput = previousLayer->dOutput;
}

bool LayerBase::writeFloat(std::ostream &stream, float value)
{
    stream.write(reinterpret_cast<char *>(&value), sizeof(value));

    return stream.good();
}

bool LayerBase::readFloat(std::istream &stream, float &value)
{
    stream.read(reinterpret_cast<char *>(&value), sizeof(value));

    return stream.good();
}

void LayerBase::updateParams(const TrainerBase &trainer) {}

void LayerBase::initAdditionalMemory(unsigned additionalMemory) {}