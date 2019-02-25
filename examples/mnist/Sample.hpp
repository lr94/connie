#ifndef CNN_SAMPLE_HPP
#define CNN_SAMPLE_HPP

#include <istream>
#include <stdexcept>

class Sample
{
public:
    friend class Dataset;

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

private:
    Sample(std::istream &dataStream, std::istream &labelStream, unsigned width, unsigned height) : w(width), h(height), data(new char[w * h]), lbl(0)
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


#endif //CNN_SAMPLE_HPP
