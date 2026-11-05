#include "image_io.hpp"

#include <memory>
#include <stdexcept>

// The stb headers hold declarations and definitions; these macros enable the definitions,
// and must appear in exactly one translation unit.
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace seg {

ImageU8 load_rgb(const std::string& path) {
    constexpr int kChannels = 3;
    int width = 0;
    int height = 0;
    int channels_in_file = 0;
    // stb allocates with malloc, so the owner must release with stbi_image_free.
    std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> pixels(
        stbi_load(path.c_str(), &width, &height, &channels_in_file, kChannels), stbi_image_free);
    if (pixels == nullptr) {
        throw std::runtime_error("failed to load " + path + ": " + stbi_failure_reason());
    }
    ImageU8 image(width, height, kChannels);
    std::copy(pixels.get(), pixels.get() + image.data.size(), image.data.begin());
    return image;
}

void save_png(const std::string& path, const ImageU8& image) {
    if (image.channels != 3 || image.width <= 0 || image.height <= 0) {
        throw std::invalid_argument("save_png: expected a non-empty 3-channel image");
    }
    const int stride_bytes = image.width * image.channels;
    if (stbi_write_png(path.c_str(), image.width, image.height, image.channels,
                       image.data.data(), stride_bytes) == 0) {
        throw std::runtime_error("failed to write " + path);
    }
}

}  // namespace seg
