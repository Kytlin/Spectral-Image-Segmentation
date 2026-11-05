#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace seg {

// A row-major image with interleaved channels that owns its samples.
// Sample (x, y, c) is stored at data[(y * width + x) * channels + c].
// Pixel index convention used everywhere in the project: p = y * width + x.
template <typename T>
struct Image {
    int width = 0;
    int height = 0;
    int channels = 0;
    std::vector<T> data;

    Image() = default;
    Image(int w, int h, int c, T fill = T{}) : width(w), height(h), channels(c) {
        if (w < 0 || h < 0 || c <= 0) {
            throw std::invalid_argument("Image: invalid dimensions");
        }
        data.assign(static_cast<std::size_t>(w) * static_cast<std::size_t>(h) *
                        static_cast<std::size_t>(c),
                    fill);
    }

    int num_pixels() const { return width * height; }

    std::size_t index(int x, int y, int c = 0) const {
        return (static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                static_cast<std::size_t>(x)) *
                   static_cast<std::size_t>(channels) +
               static_cast<std::size_t>(c);
    }
    T& at(int x, int y, int c = 0) { return data[index(x, y, c)]; }
    const T& at(int x, int y, int c = 0) const { return data[index(x, y, c)]; }
};

using ImageU8 = Image<std::uint8_t>;  // 8-bit sRGB, 3 channels
using ImageD = Image<double>;         // floating point, e.g. CIELAB (L, a, b)

}  // namespace seg
