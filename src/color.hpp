#pragma once

#include <array>
#include <cstdint>

#include "image.hpp"

namespace seg {

// CIELAB colour (L in [0, 100]; a, b roughly in [-128, 127]) relative to the D65 white point.
// Lab is used because Euclidean distance in Lab roughly tracks perceived colour difference,
// which RGB distance does not.
using Lab = std::array<double, 3>;

// 8-bit gamma-encoded sRGB value -> linear light in [0, 1]. Table lookup (256 entries,
// computed once), so converting an image costs no pow() calls.
double srgb_to_linear(std::uint8_t value);

// Linear-light RGB in [0, 1] -> CIELAB (linear RGB -> XYZ -> Lab).
Lab linear_rgb_to_lab(double r, double g, double b);

// One 8-bit sRGB colour -> CIELAB. Equals linear_rgb_to_lab(srgb_to_linear(r), ...).
Lab srgb_to_lab(std::uint8_t r, std::uint8_t g, std::uint8_t b);

// CIELAB -> 8-bit sRGB, clamping out-of-gamut values. Inverse of srgb_to_lab up to rounding.
std::array<std::uint8_t, 3> lab_to_srgb(const Lab& lab);

// Whole-image conversions. The Lab image has 3 double channels (L, a, b).
ImageD rgb_to_lab(const ImageU8& rgb);
ImageU8 lab_to_rgb(const ImageD& lab);

// Linear-light RGB image (3 channels, [0, 1]) -> Lab image / 8-bit sRGB image.
ImageD linear_rgb_to_lab(const ImageD& linear);
ImageU8 linear_to_srgb(const ImageD& linear);

}  // namespace seg
