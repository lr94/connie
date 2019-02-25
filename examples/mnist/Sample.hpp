#ifndef CNN_SAMPLE_HPP
#define CNN_SAMPLE_HPP

#include <istream>
#include <stdexcept>

class Sample
{
public:
    friend class Dataset;

    unsigned width()
    {
        return w;
    }

    unsigned height()
    {
        return h;
    }

    unsigned char label()
    {
        return lbl;
    }

    char operator[](unsigned index)
    {
        return data[index];
    }

    char getPixel(unsigned x, unsigned y)
    {
        return data[w * y + x];
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
