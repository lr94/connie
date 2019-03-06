#ifndef CONNIE_SAMPLE_HPP
#define CONNIE_SAMPLE_HPP

#include <istream>
#include <algorithm>
#include <stdexcept>

class MnistSample
{
public:
    friend class MnistDataset;

    MnistSample(const MnistSample &original) : w(original.w), h(original.h), lbl(original.lbl), data(new char[w * h])
    {
        unsigned size = w * h;
        std::copy(original.data, original.data + size, data);
    }

    MnistSample(MnistSample &&original) noexcept : w(original.w), h(original.h), lbl(original.lbl)
    {
        data = original.data;
        original.data = nullptr;
    }

    MnistSample &operator=(const MnistSample &source)
    {
        if (this != &source)
        {
            w = source.w;
            h = source.h;
            lbl = source.lbl;

            unsigned size = w * h;
            std::copy(source.data, source.data + size, data);
        }

        return *this;
    }

    MnistSample &operator=(MnistSample &&source) noexcept
    {
        if (this != &source)
        {
            w = source.w;
            h = source.h;
            lbl = source.lbl;

            data = source.data;
            source.data = nullptr;
        }

        return *this;
    }

    unsigned width() const
    {
        return w;
    }

    unsigned height() const
    {
        return h;
    }

    unsigned label() const
    {
        return lbl;
    }

    unsigned char operator[](unsigned index)
    {
        return static_cast<unsigned char>(data[index]);
    }

    unsigned char getPixel(unsigned x, unsigned y) const
    {
        return static_cast<unsigned char>(data[w * y + x]);
    }

    ~MnistSample()
    {
        delete[] data;
    }

private:
    MnistSample(std::istream &dataStream, std::istream &labelStream, unsigned width, unsigned height) : w(width), h(height), lbl(0), data(new char[w * h])
    {
        unsigned totPixels = width * height;
        for (unsigned i = 0; i < totPixels; i++)
        {
            dataStream.read(data + i, sizeof(*data));

            if (!dataStream.good())
                throw std::runtime_error("Error reading dataset");
        }

        labelStream.read(reinterpret_cast<char*>(&lbl), sizeof(lbl));
        if (!labelStream.good())
            throw std::runtime_error("Error reading dataset");
    }

    unsigned w, h;
    unsigned char lbl;
    char *data;
};


#endif //CONNIE_SAMPLE_HPP
