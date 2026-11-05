#pragma once

#include <string>

#include "image.hpp"

namespace seg {

// Loads a PNG/JPEG/BMP file as 3-channel 8-bit sRGB (alpha is dropped, gray is expanded).
// Throws std::runtime_error if the file can't be read or decoded.
ImageU8 load_rgb(const std::string& path);

// Writes a 3-channel 8-bit image as PNG. Throws std::runtime_error on failure.
void save_png(const std::string& path, const ImageU8& image);

}  // namespace seg
