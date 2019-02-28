#ifndef CNN_MEMORYSTREAM_HPP
#define CNN_MEMORYSTREAM_HPP
#include <istream>
#include <streambuf>

struct InputMemoryBuffer : std::streambuf
{
    InputMemoryBuffer(const unsigned char *data, size_t length)
    {
        char *ptr = reinterpret_cast<char *>(const_cast<unsigned char *>(data));
        this->setg(ptr, ptr, ptr + length);
    }
};

struct MemoryStream : virtual InputMemoryBuffer, std::istream
{
    MemoryStream(const unsigned char *data, size_t length) : InputMemoryBuffer(data, length), std::istream(static_cast<std::streambuf*>(this)) {}
};


#endif //CNN_MEMORYSTREAM_HPP
