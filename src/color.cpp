#include "color.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace seg {
namespace {

// D65 reference white in XYZ (Y normalized to 1).
constexpr double kWhiteX = 0.95047;
constexpr double kWhiteY = 1.00000;
constexpr double kWhiteZ = 1.08883;
// CIE constant: delta = 6/29. Below delta^3 the cube root is replaced by a line.
constexpr double kDelta = 6.0 / 29.0;

double srgb_decode(double c) {  // gamma-encoded [0,1] -> linear light [0,1]
    return c <= 0.04045 ? c / 12.92 : std::pow((c + 0.055) / 1.055, 2.4);
}
double srgb_encode(double c) {  // linear light -> gamma-encoded
    return c <= 0.0031308 ? 12.92 * c : 1.055 * std::pow(c, 1.0 / 2.4) - 0.055;
}
double lab_f(double t) {
    return t > kDelta * kDelta * kDelta ? std::cbrt(t) : t / (3.0 * kDelta * kDelta) + 4.0 / 29.0;
}
double lab_f_inverse(double t) {
    return t > kDelta ? t * t * t : 3.0 * kDelta * kDelta * (t - 4.0 / 29.0);
}

}  // namespace

double srgb_to_linear(std::uint8_t value) {
    // Built on first use; C++11 guarantees thread-safe initialization of function statics.
    static const std::array<double, 256> table = [] {
        std::array<double, 256> t{};
        for (std::size_t i = 0; i < t.size(); ++i) {
            t[i] = srgb_decode(static_cast<double>(i) / 255.0);
        }
        return t;
    }();
    return table[value];
}

Lab linear_rgb_to_lab(double r, double g, double b) {
    // Linear sRGB -> XYZ (D65), IEC 61966-2-1 matrix.
    const double x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b;
    const double y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b;
    const double z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b;
    const double fx = lab_f(x / kWhiteX);
    const double fy = lab_f(y / kWhiteY);
    const double fz = lab_f(z / kWhiteZ);
    return {116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)};
}

Lab srgb_to_lab(std::uint8_t r8, std::uint8_t g8, std::uint8_t b8) {
    return linear_rgb_to_lab(srgb_to_linear(r8), srgb_to_linear(g8), srgb_to_linear(b8));
}

std::array<std::uint8_t, 3> lab_to_srgb(const Lab& lab) {
    const double fy = (lab[0] + 16.0) / 116.0;
    const double fx = fy + lab[1] / 500.0;
    const double fz = fy - lab[2] / 200.0;
    const double x = kWhiteX * lab_f_inverse(fx);
    const double y = kWhiteY * lab_f_inverse(fy);
    const double z = kWhiteZ * lab_f_inverse(fz);
    // XYZ -> linear sRGB: inverse of the matrix above.
    const double linear[3] = {
        3.2404542 * x - 1.5371385 * y - 0.4985314 * z,
        -0.9692660 * x + 1.8760108 * y + 0.0415560 * z,
        0.0556434 * x - 0.2040259 * y + 1.0572252 * z,
    };
    std::array<std::uint8_t, 3> out{};
    for (int c = 0; c < 3; ++c) {
        const double encoded = srgb_encode(std::clamp(linear[c], 0.0, 1.0));
        out[static_cast<std::size_t>(c)] =
            static_cast<std::uint8_t>(std::lround(std::clamp(encoded, 0.0, 1.0) * 255.0));
    }
    return out;
}

ImageD rgb_to_lab(const ImageU8& rgb) {
    if (rgb.channels != 3) {
        throw std::invalid_argument("rgb_to_lab: expected 3 channels");
    }
    ImageD lab(rgb.width, rgb.height, 3);
    for (int y = 0; y < rgb.height; ++y) {
        for (int x = 0; x < rgb.width; ++x) {
            const Lab value = srgb_to_lab(rgb.at(x, y, 0), rgb.at(x, y, 1), rgb.at(x, y, 2));
            for (int c = 0; c < 3; ++c) {
                lab.at(x, y, c) = value[static_cast<std::size_t>(c)];
            }
        }
    }
    return lab;
}

ImageU8 lab_to_rgb(const ImageD& lab) {
    if (lab.channels != 3) {
        throw std::invalid_argument("lab_to_rgb: expected 3 channels");
    }
    ImageU8 rgb(lab.width, lab.height, 3);
    for (int y = 0; y < lab.height; ++y) {
        for (int x = 0; x < lab.width; ++x) {
            const auto value = lab_to_srgb({lab.at(x, y, 0), lab.at(x, y, 1), lab.at(x, y, 2)});
            for (int c = 0; c < 3; ++c) {
                rgb.at(x, y, c) = value[static_cast<std::size_t>(c)];
            }
        }
    }
    return rgb;
}

ImageD linear_rgb_to_lab(const ImageD& linear) {
    if (linear.channels != 3) {
        throw std::invalid_argument("linear_rgb_to_lab: expected 3 channels");
    }
    ImageD lab(linear.width, linear.height, 3);
    for (std::size_t p = 0; p < linear.data.size(); p += 3) {
        const Lab value = linear_rgb_to_lab(linear.data[p], linear.data[p + 1], linear.data[p + 2]);
        lab.data[p] = value[0];
        lab.data[p + 1] = value[1];
        lab.data[p + 2] = value[2];
    }
    return lab;
}

ImageU8 linear_to_srgb(const ImageD& linear) {
    if (linear.channels != 3) {
        throw std::invalid_argument("linear_to_srgb: expected 3 channels");
    }
    ImageU8 out(linear.width, linear.height, 3);
    for (std::size_t i = 0; i < linear.data.size(); ++i) {
        const double encoded = srgb_encode(std::clamp(linear.data[i], 0.0, 1.0));
        out.data[i] = static_cast<std::uint8_t>(std::lround(std::clamp(encoded, 0.0, 1.0) * 255.0));
    }
    return out;
}

}  // namespace seg
