#ifndef CNN_DATASET_HPP
#define CNN_DATASET_HPP

#include <vector>
#include <fstream>
#include <stdexcept>
#include "Sample.hpp"

class Dataset : public std::vector<Sample>
{
public:
    Dataset() = delete;

    Dataset(const char *dataFile, const char *labelsFile)
    {
        std::ifstream dataStream(dataFile, std::ifstream::binary);
        std::ifstream labelsStream(labelsFile, std::ifstream::binary);

        unsigned magicData = readUnsignedInteger(dataStream);
        unsigned magicLabels = readUnsignedInteger(labelsStream);

        if (magicData != 0x00000803)
            throw std::runtime_error("Error reading dataset");

        if (magicLabels != 0x00000801)
            throw std::runtime_error("Error reading dataset");

        unsigned n = readUnsignedInteger(dataStream);
        if (n != readUnsignedInteger(labelsStream))
            throw std::runtime_error("Error reading dataset");

        unsigned width = readUnsignedInteger(dataStream);
        unsigned height = readUnsignedInteger(dataStream);

        for (unsigned i = 0; i < n; i++)
            this->push_back(Sample(dataStream, labelsStream, width, height));
    }

private:
    unsigned readUnsignedInteger(std::istream &stream)
    {
        int endianessCheck = 1;
        char data[4];
        unsigned value;
        stream.read(data, sizeof(value));

        if (!stream.good())
            throw std::runtime_error("Error reading dataset");

        if (*reinterpret_cast<char *>(&endianessCheck) == 1) // If little endian
        {
            char tmp = data[0];
            data[0] = data[3];
            data[3] = tmp;
            tmp = data[1];
            data[1] = data[2];
            data[2] = tmp;
        }

        value = *reinterpret_cast<unsigned *>(data);

        return value;
    }
};


#endif //CNN_DATASET_HPP
