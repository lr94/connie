#ifndef CNN_TENSOR_HPP
#define CNN_TENSOR_HPP

#include <vector>
#include <iterator>
#include <iostream>
#include <random>

template <typename T>
class SubTensor
{
public:
    /**
     * Initializes a sliced version of the Tensor object (which could be a matrix or a vector)
     *
     * @param w             Pointer to the data
     * @param w_offset      Data offset
     * @param originalShape Shape of the Tensor object that was sliced
     */
    SubTensor(T *w, unsigned w_offset, std::vector<unsigned> originalShape)
    {
        this->w = w;
        this->w_offset = w_offset;

        std::copy(originalShape.begin() + 1, originalShape.end(), std::back_inserter(this->shape));
    }

    SubTensor operator [](unsigned i)
    {
        unsigned k = 1;

        for (auto iterator = shape.begin() + 1; iterator != shape.end(); iterator++)
            k *= *iterator;

        return SubTensor(w, w_offset + k * i, shape);
    }

    operator T&()
    {
        // TODO check shape length: it must be 0
        return w[w_offset];
    }

    SubTensor &operator =(T newValue)
    {
        // TODO check shape length: it must be 0
        w[w_offset] = newValue;

        return *this;
    }

    friend std::ostream &operator <<(std::ostream &out, SubTensor subTensor)
    {
        if (subTensor.shape.size() == 0)
        {
            out << subTensor.w[subTensor.w_offset];
            return out;
        }

        out << '{';

        unsigned dimension = subTensor.shape[0];
        for (unsigned i = 0; i < dimension; i++)
        {
            out << subTensor[i];

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
class Tensor
{
public:
    Tensor(unsigned depth, unsigned height, unsigned width)
    {
        shape = {depth, height, width};
        computeDataSize();

        w = new T[depth * height * width];
    }

    Tensor(unsigned height, unsigned width) : Tensor(1u, height, width) {}

    explicit Tensor(unsigned width) : Tensor(1u, width) {}

    Tensor(const Tensor &original)
    {
        shape = original.shape;
        computeDataSize();

        w = new T[totalSize];

        std::copy(original.w, original.w + totalSize, w);
    }

    Tensor(Tensor &&original) noexcept
    {
        shape = original.shape;
        computeDataSize();
        w = original.w;
        original.w = nullptr;
    }

    ~Tensor()
    {
        delete[] w;
    }

    static Tensor random(unsigned depth, unsigned height, unsigned width)
    {
        Tensor t(depth, height, width);

        std::default_random_engine generator;
        std::normal_distribution<T> distribution(0.0, 1.0);

        size_t size = t.size();
        for (size_t i = 0; i < size; i++)
            t.w[i] = distribution(generator);

        return t;
    }

    Tensor &operator=(const Tensor &source)
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

    Tensor &operator=(Tensor &&old) noexcept
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

    Tensor &operator=(std::initializer_list<T> list)
    {
        std::copy(list.begin(), list.end(), w);

        return *this;
    }

    Tensor &operator=(std::initializer_list<std::initializer_list<T>> list)
    {
        unsigned counter = 0;

        for (auto i = list.begin(); i != list.end(); i++)
            for (auto j = (*i).begin(); j != (*i).end(); j++)
                    w[counter++] = *j;

        return *this;
    }

    Tensor &operator=(std::initializer_list<std::initializer_list<std::initializer_list<T>>> list)
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

    SubTensor<T> operator [](unsigned i)
    {
        unsigned k = 1;

        for (auto iterator = shape.begin() + 1; iterator != shape.end(); iterator++)
            k *= *iterator;

        return SubTensor<T>(w, k * i, shape);
    }

    friend std::ostream &operator <<(std::ostream &out, Tensor &tensor)
    {
        if (tensor.shape.size() == 0)
        {
            out << tensor.w[0];
            return out;
        }

        out << '{';

        unsigned dimension = tensor.shape[0];
        for (unsigned i = 0; i < dimension; i++)
        {
            out << tensor[i];

            if (i < dimension - 1)
                out << ", ";
        }

        out << '}';
        return out;
    }

    // TODO extend and improve math operators

    friend Tensor<T> operator+(Tensor &left, const Tensor &right)
    {
        Tensor<T> result = left;

        size_t size = result.size();

        for (size_t i = 0; i < size; i++)
            result.w[i] += right.w[i];

        return result;
    }

    friend Tensor<T> operator+(const Tensor &left, const T &right)
    {
        Tensor<T> result = left;

        size_t size = result.size();

        for (size_t i = 0; i < size; i++)
            result.w[i] += right;

        return result;
    }

    friend Tensor operator+(Tensor &&left, const T &right)
    {
        Tensor<T> result = left;

        size_t size = result.size();

        for (size_t i = 0; i < size; i++)
            result.w[i] += right;

        return result;
    }

    friend Tensor operator+(const T &left, Tensor &right)
    {
        return right + left;
    }

    friend Tensor operator+(const T &left, Tensor &&right)
    {
        return right + left;
    }

    Tensor &operator+=(const Tensor &right)
    {
        size_t size = size();

        for (size_t i = 0; i < size; i++)
            w[i] += right.w[i];

        return *this;
    }

    Tensor &operator+=(const T &right)
    {
        size_t size = size();

        for (size_t i = 0; i < size; i++)
            w[i] += right;

        return *this;
    }

    friend Tensor<T> operator-(Tensor &left, const Tensor &right)
    {
        Tensor<T> result = left;

        size_t size = result.size();

        for (size_t i = 0; i < size; i++)
            result.w[i] -= right.w[i];

        return result;
    }

    friend Tensor<T> operator-(const Tensor &left, const T &right)
    {
        Tensor<T> result = left;

        size_t size = result.size();

        for (size_t i = 0; i < size; i++)
            result.w[i] -= right;

        return result;
    }

    friend Tensor operator-(Tensor &&left, const T &right)
    {
        Tensor<T> result = left;

        size_t size = result.size();

        for (size_t i = 0; i < size; i++)
            result.w[i] -= right;

        return result;
    }

    friend Tensor operator-(const T &left, Tensor &right)
    {
        return right - left;
    }

    friend Tensor operator-(const T &left, Tensor &&right)
    {
        return right - left;
    }

    Tensor &operator-=(const Tensor &right)
    {
        size_t size = size();

        for (size_t i = 0; i < size; i++)
            w[i] -= right.w[i];

        return *this;
    }

    Tensor &operator-=(const T &right)
    {
        size_t size = size();

        for (size_t i = 0; i < size; i++)
            w[i] -= right;

        return *this;
    }

    friend T operator*(const Tensor &left, const Tensor &right)
    {
        T result = 0;
        size_t dataSize = left.size();

        for (size_t i = 0; i < dataSize; i++)
            result += left.w[i] * right.w[i];

        return result;
    }

    friend Tensor<T> operator*(const Tensor &left, const T &right)
    {
        Tensor<T> result = left;

        size_t size = result.size();

        for (size_t i = 0; i < size; i++)
            result.w[i] *= right;

        return result;
    }

    friend Tensor operator*(Tensor &&left, const T &right)
    {
        Tensor<T> result = left;

        size_t size = result.size();

        for (size_t i = 0; i < size; i++)
            result.w[i] *= right;

        return result;
    }

    friend Tensor operator*(const T &left, Tensor &right)
    {
        return right * left;
    }

    friend Tensor operator*(const T &left, Tensor &&right)
    {
        return right * left;
    }

    Tensor &operator*=(const T &right)
    {
        size_t size = size();

        for (size_t i = 0; i < size; i++)
            w[i] *= right;

        return *this;
    }

    friend Tensor<T> operator/(const Tensor &left, const T &right)
    {
        Tensor<T> result = left;

        size_t size = result.size();

        for (size_t i = 0; i < size; i++)
            result.w[i] /= right;

        return result;
    }

    friend Tensor operator/(Tensor &&left, const T &right)
    {
        Tensor<T> result = left;

        size_t size = result.size();

        for (size_t i = 0; i < size; i++)
            result.w[i] /= right;

        return result;
    }

    friend Tensor operator/(const T &left, Tensor &right)
    {
        return right / left;
    }

    friend Tensor operator/(const T &left, Tensor &&right)
    {
        return right / left;
    }

    Tensor &operator/=(const T &right)
    {
        size_t size = size();

        for (size_t i = 0; i < size; i++)
            w[i] /= right;

        return *this;
    }

    // TODO try to improve operators

    Tensor convolve(Tensor &filter, unsigned stride)
    {
        unsigned resultHeight = (height() - filter.height()) / stride + 1;
        unsigned resultWidth = (width() - filter.width()) / stride + 1;

        Tensor result(1, resultHeight, resultWidth);

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
    void convolve(Tensor &result, unsigned resultLayer, Tensor &filter, unsigned stride)
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
        std::fill_n(w, size(), 0);
    }

    inline size_t size() const
    {
        return totalSize;
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

    size_t totalSize;

    void computeDataSize()
    {
        totalSize = 1;
        for (auto iterator = this->shape.begin(); iterator != this->shape.end(); iterator++)
            totalSize *= *iterator;
    }
};

#endif //CNN_TENSOR_HPP
