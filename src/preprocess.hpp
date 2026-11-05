#pragma once

#include <vector>

#include "image.hpp"

namespace seg {

struct PreprocessParams {
    // Longest side of the working image; larger images are shrunk by an integer factor.
    // 0 = keep full resolution. ~1000 px keeps detail far finer than typical superpixels
    // while bounding time and memory (a 22 MP photo would otherwise need ~530 MB as doubles).
    int max_dim = 1000;
    // Gaussian blur sigma in working-image pixels, applied in Lab. 0 = none. Around 1 helps
    // SLIC on heavy texture (gravel, scales) at the cost of slightly softer edges.
    double blur_sigma = 0.0;
};

struct Preprocessed {
    ImageD lab;      // working-resolution CIELAB image (input to SLIC and the graph)
    ImageU8 rgb;     // the same working image as 8-bit sRGB, for drawing
    int factor = 1;  // working pixel (x, y) covers original [f*x, f*x+f) x [f*y, f*y+f)
};

// Full pipeline: shrink first, convert after. Each f x f block of 8-bit sRGB pixels is
// decoded to LINEAR light (table lookup) and averaged there, which is how light physically
// mixes (averaging gamma-encoded values darkens edges). Only the small image is converted to
// Lab, and no full-resolution floating-point image is ever allocated.
Preprocessed preprocess(const ImageU8& rgb, const PreprocessParams& params);

// Smallest integer factor f >= 1 such that max(width, height) / f <= max_dim.
// max_dim <= 0 means "no limit" and returns 1.
int downsample_factor(int width, int height, int max_dim);

// Shrinks by an integer factor: each output pixel is the mean of an f x f input block.
// Output size is ceil(width / f) x ceil(height / f); partial blocks at the right/bottom
// edge average only the pixels they contain, so no input pixel is dropped.
ImageD downsample(const ImageD& image, int factor);

// Normalized 1-D Gaussian weights for offsets -r..r with r = ceil(3 sigma).
// sigma <= 0 returns {1.0} (no blur).
std::vector<double> gaussian_kernel(double sigma);

// Separable Gaussian blur (horizontal then vertical pass), clamping coordinates at the edges,
// applied to every channel independently.
ImageD gaussian_blur(const ImageD& image, double sigma);

}  // namespace seg
