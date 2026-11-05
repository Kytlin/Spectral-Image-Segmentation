#include "visualize.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

namespace seg {
namespace {

using Rgb = std::array<std::uint8_t, 3>;

void put(ImageU8& image, int x, int y, const Rgb& color) {
    if (x < 0 || y < 0 || x >= image.width || y >= image.height) {
        return;
    }
    for (int c = 0; c < 3; ++c) {
        image.at(x, y, c) = color[static_cast<std::size_t>(c)];
    }
}

// Bresenham's line algorithm.
void draw_line(ImageU8& image, int x0, int y0, int x1, int y1, const Rgb& color) {
    const int dx = std::abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    const int dy = -std::abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;
    while (true) {
        put(image, x0, y0, color);
        if (x0 == x1 && y0 == y1) {
            break;
        }
        const int e2 = 2 * err;
        if (e2 >= dy) {
            err += dy;
            x0 += sx;
        }
        if (e2 <= dx) {
            err += dx;
            y0 += sy;
        }
    }
}

std::uint8_t lerp_u8(double a, double b, double t) {
    return static_cast<std::uint8_t>(std::lround(a + (b - a) * t));
}

// Three-stop approximation of the viridis colour map: 0 -> purple, 0.5 -> teal, 1 -> yellow.
Rgb weight_color(double w) {
    w = std::clamp(w, 0.0, 1.0);
    const double lo[3] = {68, 1, 84}, mid[3] = {33, 145, 140}, hi[3] = {253, 231, 37};
    const double* from = w < 0.5 ? lo : mid;
    const double* to = w < 0.5 ? mid : hi;
    const double t = w < 0.5 ? w * 2.0 : (w - 0.5) * 2.0;
    return {lerp_u8(from[0], to[0], t), lerp_u8(from[1], to[1], t), lerp_u8(from[2], to[2], t)};
}

}  // namespace

ImageU8 draw_region_boundaries(const ImageU8& rgb, const std::vector<int>& labels,
                               std::array<std::uint8_t, 3> color) {
    if (static_cast<int>(labels.size()) != rgb.num_pixels()) {
        throw std::invalid_argument("draw_region_boundaries: labels size mismatch");
    }
    ImageU8 out = rgb;
    for (int y = 0; y < rgb.height; ++y) {
        for (int x = 0; x < rgb.width; ++x) {
            const int label = labels[static_cast<std::size_t>(y * rgb.width + x)];
            const bool right = x + 1 < rgb.width &&
                               labels[static_cast<std::size_t>(y * rgb.width + x + 1)] != label;
            const bool down = y + 1 < rgb.height &&
                              labels[static_cast<std::size_t>((y + 1) * rgb.width + x)] != label;
            if (right || down) {
                put(out, x, y, color);
            }
        }
    }
    return out;
}

ImageU8 draw_region_graph(const ImageU8& rgb, const RegionGraph& graph, int scale) {
    if (rgb.width != graph.width || rgb.height != graph.height || scale < 1) {
        throw std::invalid_argument("draw_region_graph: size mismatch or bad scale");
    }
    // Upscale (nearest neighbour) and lighten: out = 0.4 * pixel + 0.6 * white, so dark
    // regions turn mid-gray and the coloured edges stay visible everywhere.
    ImageU8 out(rgb.width * scale, rgb.height * scale, 3);
    for (int y = 0; y < out.height; ++y) {
        for (int x = 0; x < out.width; ++x) {
            const int sx = x / scale, sy = y / scale;
            const int label = graph.pixel_to_node[static_cast<std::size_t>(sy * rgb.width + sx)];
            const bool on_boundary =
                (x % scale == scale - 1 && sx + 1 < rgb.width &&
                 graph.pixel_to_node[static_cast<std::size_t>(sy * rgb.width + sx + 1)] != label) ||
                (y % scale == scale - 1 && sy + 1 < rgb.height &&
                 graph.pixel_to_node[static_cast<std::size_t>((sy + 1) * rgb.width + sx)] != label);
            for (int c = 0; c < 3; ++c) {
                const double v = 0.4 * rgb.at(sx, sy, c) + 0.6 * 255.0;
                out.at(x, y, c) = on_boundary ? std::uint8_t{150}
                                              : static_cast<std::uint8_t>(std::lround(v));
            }
        }
    }
    auto to_px = [scale](double v) { return static_cast<int>(std::lround((v + 0.5) * scale)); };
    // Draw weak edges first so strong edges end up on top where they cross.
    std::vector<const RegionEdge*> order;
    for (const RegionEdge& e : graph.edges) {
        order.push_back(&e);
    }
    std::sort(order.begin(), order.end(),
              [](const RegionEdge* l, const RegionEdge* r) { return l->weight < r->weight; });
    for (const RegionEdge* e : order) {
        const Region& a = graph.regions[static_cast<std::size_t>(e->a)];
        const Region& b = graph.regions[static_cast<std::size_t>(e->b)];
        draw_line(out, to_px(a.centroid_x), to_px(a.centroid_y), to_px(b.centroid_x),
                  to_px(b.centroid_y), weight_color(e->weight));
    }
    const Rgb red = {220, 30, 30};
    for (const Region& r : graph.regions) {
        const int cx = to_px(r.centroid_x), cy = to_px(r.centroid_y);
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                put(out, cx + dx, cy + dy, red);
            }
        }
    }
    return out;
}

}  // namespace seg
