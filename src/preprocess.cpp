#include "preprocess.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "color.hpp"

namespace seg {

int downsample_factor(int width, int height, int max_dim) {
    if (max_dim <= 0) {
        return 1;
    }
    const int largest = std::max(width, height);
    return std::max(1, (largest + max_dim - 1) / max_dim);  // ceil(largest / max_dim)
}

ImageD downsample(const ImageD& image, int factor) {
    if (factor < 1) {
        throw std::invalid_argument("downsample: factor must be >= 1");
    }
    if (factor == 1) {
        return image;
    }
    const int out_w = (image.width + factor - 1) / factor;
    const int out_h = (image.height + factor - 1) / factor;
    ImageD out(out_w, out_h, image.channels);
    for (int oy = 0; oy < out_h; ++oy) {
        for (int ox = 0; ox < out_w; ++ox) {
            const int x0 = ox * factor;
            const int y0 = oy * factor;
            const int x1 = std::min(x0 + factor, image.width);
            const int y1 = std::min(y0 + factor, image.height);
            const double count = static_cast<double>((x1 - x0) * (y1 - y0));
            for (int c = 0; c < image.channels; ++c) {
                double sum = 0.0;
                for (int y = y0; y < y1; ++y) {
                    for (int x = x0; x < x1; ++x) {
                        sum += image.at(x, y, c);
                    }
                }
                out.at(ox, oy, c) = sum / count;
            }
        }
    }
    return out;
}

std::vector<double> gaussian_kernel(double sigma) {
    if (!(sigma > 0.0)) {
        return {1.0};
    }
    const int radius = static_cast<int>(std::ceil(3.0 * sigma));
    std::vector<double> kernel(static_cast<std::size_t>(2 * radius + 1));
    double sum = 0.0;
    for (int i = -radius; i <= radius; ++i) {
        const double value = std::exp(-(i * i) / (2.0 * sigma * sigma));
        kernel[static_cast<std::size_t>(i + radius)] = value;
        sum += value;
    }
    for (double& value : kernel) {
        value /= sum;  // weights sum to 1, so a constant image stays constant
    }
    return kernel;
}

ImageD gaussian_blur(const ImageD& image, double sigma) {
    const std::vector<double> kernel = gaussian_kernel(sigma);
    if (kernel.size() == 1) {
        return image;
    }
    const int radius = static_cast<int>(kernel.size() / 2);
    ImageD horizontal(image.width, image.height, image.channels);
    for (int y = 0; y < image.height; ++y) {
        for (int x = 0; x < image.width; ++x) {
            for (int c = 0; c < image.channels; ++c) {
                double sum = 0.0;
                for (int i = -radius; i <= radius; ++i) {
                    const int sx = std::clamp(x + i, 0, image.width - 1);
                    sum += kernel[static_cast<std::size_t>(i + radius)] * image.at(sx, y, c);
                }
                horizontal.at(x, y, c) = sum;
            }
        }
    }
    ImageD out(image.width, image.height, image.channels);
    for (int y = 0; y < image.height; ++y) {
        for (int x = 0; x < image.width; ++x) {
            for (int c = 0; c < image.channels; ++c) {
                double sum = 0.0;
                for (int i = -radius; i <= radius; ++i) {
                    const int sy = std::clamp(y + i, 0, image.height - 1);
                    sum += kernel[static_cast<std::size_t>(i + radius)] * horizontal.at(x, sy, c);
                }
                out.at(x, y, c) = sum;
            }
        }
    }
    return out;
}

Preprocessed preprocess(const ImageU8& rgb, const PreprocessParams& params) {
    if (rgb.channels != 3 || rgb.width <= 0 || rgb.height <= 0) {
        throw std::invalid_argument("preprocess: expected a non-empty 3-channel image");
    }
    const int f = downsample_factor(rgb.width, rgb.height, params.max_dim);
    const int out_w = (rgb.width + f - 1) / f;
    const int out_h = (rgb.height + f - 1) / f;

    // Fused decode + box average: read 8-bit pixels, accumulate linear light per block.
    ImageD linear(out_w, out_h, 3);
    for (int oy = 0; oy < out_h; ++oy) {
        const int y0 = oy * f, y1 = std::min(y0 + f, rgb.height);
        for (int ox = 0; ox < out_w; ++ox) {
            const int x0 = ox * f, x1 = std::min(x0 + f, rgb.width);
            double sum[3] = {0.0, 0.0, 0.0};
            for (int y = y0; y < y1; ++y) {
                for (int x = x0; x < x1; ++x) {
                    for (int c = 0; c < 3; ++c) {
                        sum[c] += srgb_to_linear(rgb.at(x, y, c));
                    }
                }
            }
            const double inv = 1.0 / static_cast<double>((x1 - x0) * (y1 - y0));
            for (int c = 0; c < 3; ++c) {
                linear.at(ox, oy, c) = sum[c] * inv;
            }
        }
    }

    Preprocessed out;
    out.factor = f;
    out.rgb = linear_to_srgb(linear);
    out.lab = gaussian_blur(linear_rgb_to_lab(linear), params.blur_sigma);
    return out;
}

}  // namespace seg
