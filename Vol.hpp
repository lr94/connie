#ifndef CNN_VOL_HPP
#define CNN_VOL_HPP

#include <vector>
#include <iterator>
#include <iostream>
#include <random>

template <typename T>
class SubVol
{
public:
    /**
     * Initializes a sliced version of the Vol object (which could be a matrix or a vector)
     *
     * @param w             Pointer to the data
     * @param w_offset      Data offset
     * @param originalShape Shape of the Vol object that was sliced
     */
    SubVol(T *w, unsigned w_offset, std::vector<unsigned> originalShape)
    {
        this->w = w;
        this->w_offset = w_offset;

        std::copy(originalShape.begin() + 1, originalShape.end(), std::back_inserter(this->shape));
    }

    SubVol operator [](unsigned i)
    {
        unsigned k = 1;

        for (auto iterator = shape.begin() + 1; iterator != shape.end(); iterator++)
            k *= *iterator;

        return SubVol(w, w_offset + k * i, shape);
    }

    operator T&()
    {
        // TODO check shape length: it must be 0
        return w[w_offset];
    }

    SubVol &operator =(T newValue)
    {
        // TODO check shape length: it must be 0
        w[w_offset] = newValue;

        return *this;
    }

    friend std::ostream &operator <<(std::ostream &out, SubVol subVol)
    {
        if (subVol.shape.size() == 0)
        {
            out << subVol.w[subVol.w_offset];
            return out;
        }

        out << '{';

        unsigned dimension = subVol.shape[0];
        for (unsigned i = 0; i < dimension; i++)
        {
            out << subVol[i];

            if (i < dimension - 1)
                out << ", ";
        }

        out << '}';
        return out;
    }

private:
    T *w;
    unsigned w_offset;

    std::vector<unsigned> shape;
};

template <typename T = float>
class Vol
{
public:
    Vol(unsigned depth, unsigned height, unsigned width)
    {
        shape = {depth, height, width};
        computeDataSize();

        w = new T[depth * height * width];
    }

    Vol(unsigned height, unsigned width) : Vol(1u, height, width) {}

    explicit Vol(unsigned width) : Vol(1u, width) {}

    Vol(const Vol &original)
    {
        shape = original.shape;
        computeDataSize();

        size_t size = size;
        w = new T[size];

        std::copy(original.w, original.w + size, w);
    }

    Vol(Vol &&original) noexcept
    {
        shape = original.shape;
        computeDataSize();
        w = original.w;
        original.w = nullptr;
    }

    ~Vol()
    {
        delete[] w;
    }

    static Vol random(unsigned depth, unsigned height, unsigned width)
    {
        Vol vol(depth, height, width);

        std::default_random_engine generator;
        std::normal_distribution<T> distribution(0.0, 1.0);

        size_t size = vol.getDataSize();
        for (size_t i = 0; i < size; i++)
            vol.w[i] = distribution(generator);

        return vol;
    }

    Vol &operator=(const Vol &source)
    {
        if (this != &source)
        {
            shape = source.shape;
            computeDataSize();

            unsigned size = 1;
            for (auto iterator = this->shape.begin(); iterator != this->shape.end(); iterator++)
                size *= *iterator;
            delete[] w;
            w = new T[size];

            std::copy(source.w, source.w + size, w);
        }

        return *this;
    }

    Vol &operator=(Vol &&old) noexcept
    {
        if (this != &old)
        {
            shape = old.shape;
            computeDataSize();

            delete[] w;
            w = old.w;
            old.w = nullptr;
        }

        return *this;
    }

    Vol &operator=(std::initializer_list<T> list)
    {
        std::copy(list.begin(), list.end(), w);

        return *this;
    }

    Vol &operator=(std::initializer_list<std::initializer_list<T>> list)
    {
        unsigned counter = 0;

        for (auto i = list.begin(); i != list.end(); i++)
            for (auto j = (*i).begin(); j != (*i).end(); j++)
                    w[counter++] = *j;

        return *this;
    }

    Vol &operator=(std::initializer_list<std::initializer_list<std::initializer_list<T>>> list)
    {
        unsigned counter = 0;

        for (auto i = list.begin(); i != list.end(); i++)
            for (auto j = (*i).begin(); j != (*i).end(); j++)
                for (auto k = (*j).begin(); k != (*j).end(); k++)
                    w[counter++] = *k;

        return *this;
    }

    inline unsigned depth() const
    {
        return shape[0];
    }

    inline unsigned height() const
    {
        return shape[1];
    }

    inline unsigned width() const
    {
        return shape[2];
    }

    SubVol<T> operator [](unsigned i)
    {
        unsigned k = 1;

        for (auto iterator = shape.begin() + 1; iterator != shape.end(); iterator++)
            k *= *iterator;

        return SubVol<T>(w, k * i, shape);
    }

    friend std::ostream &operator <<(std::ostream &out, Vol &vol)
    {
        if (vol.shape.size() == 0)
        {
            out << vol.w[0];
            return out;
        }

        out << '{';

        unsigned dimension = vol.shape[0];
        for (unsigned i = 0; i < dimension; i++)
        {
            out << vol[i];

            if (i < dimension - 1)
                out << ", ";
        }

        out << '}';
        return out;
    }

    // TODO extend and improve math operators

    friend Vol<T> operator+(Vol &left, const Vol &right)
    {
        Vol<T> result = left;

        size_t size = result.getDataSize();

        for (size_t i = 0; i < size; i++)
            result.w[i] += right.w[i];

        return result;
    }

    friend Vol<T> operator+(const Vol &left, const T &right)
    {
        Vol<T> result = left;

        size_t size = result.getDataSize();

        for (size_t i = 0; i < size; i++)
            result.w[i] += right;

        return result;
    }

    friend Vol operator+(Vol &&left, const T &right)
    {
        Vol<T> result = left;

        size_t size = result.getDataSize();

        for (size_t i = 0; i < size; i++)
            result.w[i] += right;

        return result;
    }

    friend Vol operator+(const T &left, Vol &right)
    {
        return right + left;
    }

    friend Vol operator+(const T &left, Vol &&right)
    {
        return right + left;
    }

    Vol &operator+=(const Vol &right)
    {
        size_t size = getDataSize();

        for (size_t i = 0; i < size; i++)
            w[i] += right.w[i];

        return *this;
    }

    Vol &operator+=(const T &right)
    {
        size_t size = getDataSize();

        for (size_t i = 0; i < size; i++)
            w[i] += right;

        return *this;
    }

    friend Vol<T> operator-(Vol &left, const Vol &right)
    {
        Vol<T> result = left;

        size_t size = result.getDataSize();

        for (size_t i = 0; i < size; i++)
            result.w[i] -= right.w[i];

        return result;
    }

    friend Vol<T> operator-(const Vol &left, const T &right)
    {
        Vol<T> result = left;

        size_t size = result.getDataSize();

        for (size_t i = 0; i < size; i++)
            result.w[i] -= right;

        return result;
    }

    friend Vol operator-(Vol &&left, const T &right)
    {
        Vol<T> result = left;

        size_t size = result.getDataSize();

        for (size_t i = 0; i < size; i++)
            result.w[i] -= right;

        return result;
    }

    friend Vol operator-(const T &left, Vol &right)
    {
        return right - left;
    }

    friend Vol operator-(const T &left, Vol &&right)
    {
        return right - left;
    }

    Vol &operator-=(const Vol &right)
    {
        size_t size = getDataSize();

        for (size_t i = 0; i < size; i++)
            w[i] -= right.w[i];

        return *this;
    }

    Vol &operator-=(const T &right)
    {
        size_t size = getDataSize();

        for (size_t i = 0; i < size; i++)
            w[i] -= right;

        return *this;
    }

    friend T operator*(const Vol &left, const Vol &right)
    {
        T result = 0;
        size_t dataSize = left.getDataSize();

        for (size_t i = 0; i < dataSize; i++)
            result += left.w[i] * right.w[i];

        return result;
    }

    // TODO implement +=, -, -=, *=, /, /=

    Vol convolve(Vol &filter, unsigned stride)
    {
        unsigned resultHeight = (height() - filter.height()) / stride + 1;
        unsigned resultWidth = (width() - filter.width()) / stride + 1;

        Vol result(1, resultHeight, resultWidth);

        convolve(result, 0, filter, stride);

        return result;
    }

    /**
     * Performs a convolution and stores the resulting feature map in the specified layer of result
     *
     * @param result
     * @param resultLayer
     * @param filter
     * @param stride
     */
    void convolve(Vol &result, unsigned resultLayer, Vol &filter, unsigned stride)
    {
        unsigned resultHeight = (height() - filter.height()) / stride + 1;
        unsigned resultWidth = (width() - filter.width()) / stride + 1;

        unsigned filterDepth = filter.depth();
        unsigned filterHeight = filter.height();
        unsigned filterWidth = filter.width();

        for (unsigned resultY = 0; resultY < resultHeight; resultY++)
            for (unsigned resultX = 0; resultX < resultWidth; resultX++)
                result.set(resultLayer, resultY, resultX, 0);

        for (unsigned layer = 0; layer < filterDepth; layer++)
        {
            for (unsigned resultY = 0, sourceTLCornerY = 0; resultY < resultHeight; resultY++, sourceTLCornerY += stride)
            {
                for (unsigned resultX = 0, sourceTLCornerX = 0; resultX < resultWidth; resultX++, sourceTLCornerX += stride)
                {
                    T sum = 0;

                    for (unsigned filterY = 0; filterY < filterHeight; filterY++)
                    {
                        for (unsigned filterX = 0; filterX < filterWidth; filterX++)
                        {
                            unsigned sourceX = sourceTLCornerX + filterX;
                            unsigned sourceY = sourceTLCornerY + filterY;

                            sum += get(layer, sourceY, sourceX) * filter.get(layer, filterY, filterX);
                        }
                    }

                    result.addAt(resultLayer, resultY, resultX, sum);
                }
            }
        }
    }

    void zero()
    {
        std::fill_n(w, getDataSize(), 0);
    }

    inline size_t getDataSize() const
    {
        return size;
    }

    inline T get(unsigned layer, unsigned row, unsigned column) const
    {
        unsigned layerSize = width() * height(); // Layer size
        unsigned index = layer * layerSize + row * width() + column;
        return w[index];
    }

    inline void set(unsigned layer, unsigned row, unsigned column, T value)
    {
        unsigned layerSize = width() * height(); // Layer size
        unsigned index = layer * layerSize + row * width() + column;
        w[index] = value;
    }

    inline void addAt(unsigned layer, unsigned row, unsigned column, T value)
    {
        unsigned layerSize = width() * height(); // Layer size
        unsigned index = layer * layerSize + row * width() + column;
        w[index] += value;
    }

    inline T get(unsigned index) const
    {
        return w[index];
    }

    inline void set(unsigned index, T value)
    {
        w[index] = value;
    }

    inline void addAt(unsigned index, T value)
    {
        w[index] += value;
    }

private:
    std::vector<unsigned> shape; // For possible future extension to generic n-dimensional tensors
    T *w;

    size_t size;

    void computeDataSize()
    {
        size = 1;
        for (auto iterator = this->shape.begin(); iterator != this->shape.end(); iterator++)
            size *= *iterator;
    }
};

#endif //CNN_VOL_HPP
